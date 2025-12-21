"""
CTM Synchronization Pair Initialization (nonlinear feature map, linear kernel)

Goal
----
We want to initialize CTM “random pairing” indices (left,right) for each role
(action / out) using *data-driven* structure, but with:
  - no task-specific knobs,
  - stable behavior across tasks.

Mental model
------------
Let z(x,t) ∈ R^D be the CTM activated state at internal tick t for an input x.
We collect representative states from a short warmup:

  - z★ := state at the tick with maximum step-to-step change (proxy for “most dynamic”)
  - zT := final tick state (“settled”)

We build a matrix X whose rows are standardized samples of a transformed state:

  x_row = standardize( g( rms_norm(z) ) )

where:
  - rms_norm makes each row comparable (scale-stable)
  - standardize (online) makes features comparable (dimension-wise)
  - g(·) is a *nonlinear feature map* applied elementwise.

Then, we compute a low-rank approximation to the (linear) Gram proxy:

  K ≈ X^T X   (DxD)
  X ≈ U diag(S) V^T    (SVD)
  => K ≈ V diag(S^2) V^T

Neuron “importance” comes from leverage scores:
  lev_i = ||V[i,:]||^2

We sample a candidate set C (size M ≪ D) using p_i (tempered leverage),
build a small |K_C|, and pick J edges (pairs) by greedy highest-score edges
under a degree cap.

Nonlinearity choice
-------------------
We default to a robust, universal feature map:

  g(x) = sign(x) * sqrt(|x| + eps)

Why: it compresses heavy tails *without* hard saturation, preserves sign,
and tends to be stable across tasks with different activation regimes
(mazes, image-like tasks, etc.). It is a classic variance-stabilizing transform.

Where it lives: inside the reservoir ingestion, *before* standardization.

Practical notes
---------------
- Warmup runs under torch.inference_mode() and disables dropout while preserving BN batch stats,
  so it measures realistic forward activations without training-side noise or BN drift.
- This module does not compute loss. It estimates statistics from forward dynamics only.
- We keep memory bounded with a reservoir sampler.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


Tensor = torch.Tensor
FeatureMap = Literal["identity", "signed_sqrt", "tanh", "abs", "square"]


# ---------------------------------------------------------------------------
# 1) Warmup forward mode (RNG-safe, BN-safe)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def preserve_rng_state(enabled: bool = True):
    """Preserve and restore python/numpy/torch (+cuda) RNG.

    Ensures pair initialization does not perturb training randomness streams.
    """
    if not enabled:
        yield
        return
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)


@contextlib.contextmanager
def batchnorm_use_batch_stats_no_update(model: nn.Module):
    """Disable dropout but keep BN batch stats; do not update BN running stats."""
    saved_bn: List[Tuple[nn.Module, Dict[str, Any]]] = []
    model_training = model.training
    try:
        model.eval()  # disables dropout
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                st: Dict[str, Any] = {"training": m.training, "momentum": m.momentum}
                if getattr(m, "running_mean", None) is not None:
                    st["running_mean"] = m.running_mean.detach().clone()
                if getattr(m, "running_var", None) is not None:
                    st["running_var"] = m.running_var.detach().clone()
                if getattr(m, "num_batches_tracked", None) is not None:
                    st["num_batches_tracked"] = m.num_batches_tracked.detach().clone()
                saved_bn.append((m, st))

                m.train(True)      # BN uses batch stats
                m.momentum = 0.0   # prevent running-stat drift
        yield
    finally:
        for m, st in saved_bn:
            m.train(st["training"])
            m.momentum = st["momentum"]
            if "running_mean" in st:
                m.running_mean.copy_(st["running_mean"])
            if "running_var" in st:
                m.running_var.copy_(st["running_var"])
            if "num_batches_tracked" in st:
                m.num_batches_tracked.copy_(st["num_batches_tracked"])
        model.train(model_training)


# ---------------------------------------------------------------------------
# 2) Tick representative snapshots
# ---------------------------------------------------------------------------

@dataclass
class TickSnapshots:
    z_star: Optional[Tensor]
    z_last: Optional[Tensor]
    t_star: Optional[int]
    t_last: Optional[int]


class TickRepresentativeCollector:
    """Pick z★ (max change) and zT (final) without storing all ticks."""
    def __init__(self):
        self.prev: Optional[Tensor] = None
        self.best_delta: float = -1.0
        self.best_z: Optional[Tensor] = None
        self.last_z: Optional[Tensor] = None
        self.best_tick: Optional[int] = None
        self.last_tick: Optional[int] = None

    def reset(self):
        self.prev = None
        self.best_delta = -1.0
        self.best_z = None
        self.last_z = None
        self.best_tick = None
        self.last_tick = None

    def __call__(self, tick_idx: int, activated_state: Tensor):
        z = activated_state.detach()
        self.last_z = z
        self.last_tick = tick_idx
        if self.prev is not None:
            delta = (z - self.prev).norm(dim=1).mean().item()
            if delta > self.best_delta:
                self.best_delta = delta
                self.best_z = z
                self.best_tick = tick_idx
        self.prev = z

    def snapshots(self) -> TickSnapshots:
        z_star = self.best_z if self.best_z is not None else self.last_z
        t_star = self.best_tick if self.best_tick is not None else self.last_tick
        return TickSnapshots(z_star=z_star, z_last=self.last_z, t_star=t_star, t_last=self.last_tick)


# ---------------------------------------------------------------------------
# 3) Reservoir + online standardization + Tier-0 feature map
# ---------------------------------------------------------------------------

def _rms_normalize_rows(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(z * z, axis=1, keepdims=True) + eps)
    return z / rms


def apply_feature_map(z: np.ndarray, kind: FeatureMap, eps: float = 1e-8) -> np.ndarray:
    """Elementwise feature map g(z). Keeps shape (B,D)."""
    if kind == "identity":
        return z
    if kind == "signed_sqrt":
        return np.sign(z) * np.sqrt(np.abs(z) + eps)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "abs":
        return np.abs(z)
    if kind == "square":
        return z * z
    raise ValueError(f"Unknown feature_map: {kind}")


class Reservoir:
    """Bounded-memory activation store with online standardization."""
    def __init__(self, d: int, capacity: int, *, clip: float = 5.0, feature_map: FeatureMap = "signed_sqrt"):
        self.d = int(d)
        self.capacity = int(capacity)
        self.clip = float(clip)
        self.feature_map: FeatureMap = feature_map

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.n_seen = 0
        self.size = 0
        self.data = np.empty((self.capacity, self.d), dtype=np.float32)

    def add_batch(self, z_torch: Tensor):
        if z_torch is None or z_torch.numel() == 0:
            return
        z = z_torch.detach().to(device="cpu", dtype=torch.float32).numpy()  # (B,D)
        z = _rms_normalize_rows(z)
        z = apply_feature_map(z, self.feature_map)

        self.scaler.partial_fit(z)

        b = z.shape[0]
        # Fill
        if self.size < self.capacity:
            n_fill = min(self.capacity - self.size, b)
            self.data[self.size:self.size + n_fill] = z[:n_fill]
            self.size += n_fill
            self.n_seen += n_fill
            z = z[n_fill:]
        if z.size == 0:
            return

        # Replacement (vectorized uniform reservoir)
        br = z.shape[0]
        n_seen_t = self.n_seen + np.arange(1, br + 1, dtype=np.int64)
        j = (np.random.rand(br) * n_seen_t.astype(np.float64)).astype(np.int64)
        mask = j < self.capacity
        if np.any(mask):
            self.data[j[mask]] = z[mask]
        self.n_seen += br

    def finalize_X(self) -> np.ndarray:
        """Return standardized, clipped reservoir matrix X (N,D)."""
        if self.size < 2:
            return np.empty((0, self.d), dtype=np.float32)
        mu = self.scaler.mean_.astype(np.float32, copy=False)
        std = np.sqrt(self.scaler.var_.astype(np.float32, copy=False) + 1e-12)
        X = (self.data[:self.size] - mu[None, :]) / (std[None, :] + 1e-8)
        X = np.clip(X, -self.clip, self.clip)
        return X.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 4) Low-rank Gram proxy -> leverage-tempered importance p
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LowRank:
    V: np.ndarray      # (D,k_eff)
    S: np.ndarray      # (k_eff,)
    p: np.ndarray      # (D,)
    eta: float         # energy ratio
    k_eff: int


def lowrank_from_X(X: np.ndarray, d: int, k: int) -> LowRank:
    if X.size == 0 or X.shape[0] < 2:
        p = np.full((d,), 1.0 / d, dtype=np.float32)
        return LowRank(V=np.zeros((d, 0), np.float32), S=np.zeros((0,), np.float32), p=p, eta=0.0, k_eff=0)

    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    k_eff = int(min(k, d, max(1, min(N - 1, k))))

    # Deterministic randomized SVD is preferred for reproducibility and testing.
    _, S, Vt = randomized_svd(Xc, n_components=k_eff, n_iter=2, random_state=0)
    V = Vt.T.astype(np.float32, copy=False)
    S = S.astype(np.float32, copy=False)

    lev = np.sum(V * V, axis=1).astype(np.float64)
    lev_sum = float(np.sum(lev))
    lev = (lev / lev_sum) if lev_sum > 0 else np.full((d,), 1.0 / d, dtype=np.float64)

    fro2 = float(np.sum(Xc * Xc))
    top2 = float(np.sum(S * S))
    eta = float(top2 / (fro2 + 1e-8))
    alpha = float(np.clip(1.0 - eta, 0.0, 1.0))

    p = (1.0 - alpha) * lev + alpha * (1.0 / d)
    p = np.maximum(p, 1e-12)
    p = (p / np.sum(p)).astype(np.float32, copy=False)

    return LowRank(V=V, S=S, p=p, eta=eta, k_eff=k_eff)


# ---------------------------------------------------------------------------
# 5) Candidate Gram + greedy degree-capped edge selection
# ---------------------------------------------------------------------------

def n_triangle(J: int) -> int:
    """Smallest n such that n(n-1)/2 >= J (i.e., J distinct non-self edges)."""
    J = int(max(0, J))
    if J == 0:
        return 0
    # Solve n(n-1)/2 >= J  =>  n^2 - n - 2J >= 0
    return int(math.ceil((1.0 + math.sqrt(1.0 + 8.0 * J)) / 2.0))


def sample_without_replacement(p: np.ndarray, m: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    p = p / np.sum(p)
    m = int(min(m, p.shape[0]))
    return np.random.choice(p.shape[0], size=m, replace=False, p=p).astype(np.int64)


def gram_abs_from_factors(V: np.ndarray, S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    A = (V[idx, :] * S[None, :]).astype(np.float32, copy=False)  # (M,k)
    G = np.abs(A @ A.T).astype(np.float32, copy=False)          # (M,M)
    np.fill_diagonal(G, 0.0)
    return G


def greedy_degree_capped_edges(W: np.ndarray, J: int, d_max: int) -> List[Tuple[int, int]]:
    """Deterministic greedy: accept largest scores under per-node degree cap."""
    M = W.shape[0]
    if M <= 1 or J <= 0:
        return []
    tri_i, tri_j = np.triu_indices(M, k=1)
    scores = W[tri_i, tri_j]
    order = np.argsort(scores)[::-1]

    deg = np.zeros((M,), dtype=np.int32)
    out: List[Tuple[int, int]] = []
    for k in order:
        if len(out) >= J:
            break
        if scores[k] <= 0:
            break
        i = int(tri_i[k])
        j = int(tri_j[k])
        if deg[i] >= d_max or deg[j] >= d_max:
            continue
        out.append((i, j))
        deg[i] += 1
        deg[j] += 1
    return out


def participation_ratio(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    q = np.maximum(q, 1e-12)
    q = q / np.sum(q)
    return float(1.0 / np.sum(q * q))


def pairs_from_lowrank(
    factors: LowRank,
    *,
    J: int,
    n_self: int,
    d: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Single universal policy (same for action/out).

    Steps:
      - Candidate set C ~ p (size M)
      - Build |K_C| from low-rank proxy
      - Compute hubness q from row-sums of |K_C|
      - Greedy select J edges under a degree cap

    This avoids role-specific coverage logic; hubness in the kernel decides.
    """
    J = int(J)
    d = int(d)
    n_self = int(min(n_self, max(0, J - 1)))

    if factors.k_eff <= 0:
        J = int(J)
        d = int(d)
        n_self = int(min(n_self, max(0, J - 1)))

        left_pairs: List[int] = []
        right_pairs: List[int] = []
        used = set()

        # Self-prefix: deterministic-ish even in fallback
        if n_self > 0:
            # factors.p exists even in fallback in your lowrank_from_X; if not, use uniform.
            p = getattr(factors, "p", None)
            if p is None or len(p) != d:
                top = np.arange(d, dtype=np.int64)[:n_self]
            else:
                top = np.argsort(p)[::-1][:n_self].astype(np.int64)

            for idx in top.tolist():
                left_pairs.append(int(idx))
                right_pairs.append(int(idx))
                used.add((int(idx), int(idx)))

        # Fill remaining with uniform non-self pairs when possible, avoiding duplicates.
        while len(left_pairs) < J:
            i = int(np.random.randint(0, d))
            j = int(np.random.randint(0, d))

            # Avoid additional self-pairs beyond the explicit prefix.
            if d > 1 and i == j:
                j = (i + 1) % d
            # If d == 1, self-pairs are unavoidable; allow them to terminate.
            if (i, j) in used:
                continue

            used.add((i, j))
            left_pairs.append(i)
            right_pairs.append(j)

        left = np.asarray(left_pairs[:J], dtype=np.int64)
        right = np.asarray(right_pairs[:J], dtype=np.int64)

        return left, right, {"mode": "uniform_fallback", "self": int(n_self)}


    M = int(min(M, d))
    C = sample_without_replacement(factors.p, M)  # indices in [0..D)
    G = gram_abs_from_factors(factors.V[:, :factors.k_eff], factors.S[:factors.k_eff], C)

    s = np.sum(G, axis=1).astype(np.float64)
    s = np.maximum(s, 1e-12)
    q = (s / np.sum(s)).astype(np.float32, copy=False)
    n_eff = participation_ratio(q)

    # Score edges by |K_ij| weighted by sqrt(q_i q_j)
    sqrt_q = np.sqrt(q.astype(np.float64) + 1e-12)
    W = (G * (sqrt_q[:, None] * sqrt_q[None, :]).astype(np.float32)).astype(np.float32, copy=False)
    np.fill_diagonal(W, 0.0)

    left_pairs: List[int] = []
    right_pairs: List[int] = []
    used = set()

    # self-pairs: top-q candidates (mapped to global)
    if n_self > 0:
        top_local = np.argsort(q)[::-1][:n_self]
        top_global = C[top_local]
        for idx in top_global.tolist():
            left_pairs.append(int(idx))
            right_pairs.append(int(idx))
            used.add((int(idx), int(idx)))

    J_nonself = J - n_self
    n_tri = n_triangle(J_nonself)
    # Degree cap: a single, interpretable cap derived from a dense-equivalent support size.
    # If the kernel is hubbier, greedy will naturally concentrate; if diffuse, it spreads.
    n_target = int(min(M, max(1, n_tri)))
    d_max = int(math.ceil(2 * J_nonself / max(1, n_target)))

    edges = greedy_degree_capped_edges(W, J_nonself, d_max)
    for (iC, jC) in edges:
        i = int(C[iC])
        j = int(C[jC])
        if (i, j) in used or i == j:
            continue
        used.add((i, j))
        left_pairs.append(i)
        right_pairs.append(j)

    # Guaranteed fill if greedy under-fills (rare but safe)
    while len(left_pairs) < J:
        i = int(np.random.randint(0, d))
        j = int(np.random.randint(0, d))
        # Avoid additional self-pairs beyond the explicit n_self prefix.
        if d > 1 and i == j:
            j = (i + 1) % d
        # If d == 1, self-pairs are unavoidable; allow them to prevent infinite loop.
        if (i, j) in used:
            continue
        used.add((i, j))
        left_pairs.append(i)
        right_pairs.append(j)

    diag = {
        "mode": "lowrank_greedy",
        "M": int(M),
        "self": int(n_self),
        "n_eff": float(n_eff),
        "n_tri": int(n_tri),
        "d_max": int(d_max),
    }
    return np.asarray(left_pairs[:J], np.int64), np.asarray(right_pairs[:J], np.int64), diag


# ---------------------------------------------------------------------------
# 6) Entrypoint: warmup -> reservoirs -> low-rank -> pairs -> write into model
# ---------------------------------------------------------------------------

def _default_batch_to_x(batch: Any) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch)}. Provide batch_to_x.")


def initialize_ctm_pairs(
    model: nn.Module,
    dataloader: Iterable,
    *,
    warmup_batches: int,
    n_random_pairing_self: int,
    feature_map: FeatureMap = "signed_sqrt",
    device: Optional[torch.device | str] = None,
    batch_to_x: Optional[Callable[[Any], Tensor]] = None,
    preserve_seed: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Task-agnostic CTM pair initialization (same policy for action and out)."""
    if batch_to_x is None:
        batch_to_x = _default_batch_to_x
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    d = int(getattr(model, "d_model"))
    J_action = int(getattr(model, "n_synch_action"))
    J_out = int(getattr(model, "n_synch_out"))

    # bounded compute knobs (not task knobs)
    k = int(min(256, d))
    n_res = int(min(32768, 32 * k))
    M = int(min(d, 4 * k))

    res_action = Reservoir(d=d, capacity=n_res, clip=5.0, feature_map=feature_map)
    res_out = Reservoir(d=d, capacity=n_res, clip=5.0, feature_map=feature_map)

    collector = TickRepresentativeCollector()
    tstars: List[int] = []

    rng_ctx = preserve_rng_state(preserve_seed)
    bn_ctx = batchnorm_use_batch_stats_no_update(model)
    pbar = tqdm(range(int(warmup_batches)), desc="CTM Pair Init Warmup") if show_progress else range(int(warmup_batches))

    with rng_ctx, bn_ctx, torch.inference_mode():
        it = iter(dataloader)
        for _ in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)

            x = batch_to_x(batch).to(device, non_blocking=True)

            # Break circularity: resample action pairs during warmup if available.
            if hasattr(model, "resample_action_pairs_"):
                model.resample_action_pairs_(int(n_random_pairing_self))

            collector.reset()
            _ = model(x, track=False, callback=collector)
            snaps = collector.snapshots()

            if snaps.t_star is not None:
                tstars.append(int(snaps.t_star))
            if snaps.z_star is None or snaps.z_last is None:
                continue

            # action uses z★; out uses z★ and zT
            res_action.add_batch(snaps.z_star)
            res_out.add_batch(snaps.z_star)
            res_out.add_batch(snaps.z_last)

    X_action = res_action.finalize_X()
    X_out = res_out.finalize_X()

    factors_action = lowrank_from_X(X_action, d=d, k=k)
    factors_out = lowrank_from_X(X_out, d=d, k=k)

    left_a, right_a, diag_a = pairs_from_lowrank(
        factors_action, J=J_action, n_self=n_random_pairing_self, d=d, M=M
    )
    left_o, right_o, diag_o = pairs_from_lowrank(
        factors_out, J=J_out, n_self=n_random_pairing_self, d=d, M=M
    )

    model.set_neuron_pairs(
        "action",
        torch.from_numpy(left_a).to(device=device),
        torch.from_numpy(right_a).to(device=device),
    )
    model.set_neuron_pairs(
        "out",
        torch.from_numpy(left_o).to(device=device),
        torch.from_numpy(right_o).to(device=device),
    )

    t_arr = np.asarray(tstars, dtype=np.float32) if tstars else np.asarray([0.0], dtype=np.float32)
    return {
        "feature_map": feature_map,
        "warmup_batches": int(warmup_batches),
        "reservoir_capacity": int(n_res),
        "reservoir_size_action": int(res_action.size),
        "reservoir_seen_action": int(res_action.n_seen),
        "reservoir_size_out": int(res_out.size),
        "reservoir_seen_out": int(res_out.n_seen),
        "t_star_mean": float(np.mean(t_arr)),
        "t_star_std": float(np.std(t_arr)),
        "action": {"J": int(J_action), "k_eff": int(factors_action.k_eff), "eta": float(factors_action.eta), **diag_a},
        "out": {"J": int(J_out), "k_eff": int(factors_out.k_eff), "eta": float(factors_out.eta), **diag_o},
    }
