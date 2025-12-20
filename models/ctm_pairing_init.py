from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# 1) Warmup forward mode
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def preserve_rng_state(enabled: bool = True):
    """Preserve and restore RNG states for python/numpy/torch (+cuda).

    Rationale:
      Pair initialization should not accidentally perturb training randomness
      (data augmentation, dropout, etc.). This context keeps init "side-effect-free"
      with respect to RNG streams when enabled.
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
    """Warmup mode that avoids polluting correlations with dropout but keeps BN realistic.

    Goal:
      Warmup should approximate training-time forward activations (BN batch stats),
      but must not mutate BN running_mean/var (to remain minimally invasive).

    Mechanism:
      - set model.eval() to disable dropout and other train-time stochasticity
      - force BN modules into train mode so they use batch stats
      - set BN momentum=0 to avoid running-stat drift
      - snapshot/restore BN buffers and training flags
    """
    saved_bn: List[Tuple[nn.Module, Dict[str, Any]]] = []
    model_training = model.training
    try:
        model.eval()
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
                m.train(True)
                m.momentum = 0.0
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
# 2) Tick representative snapshots: Z_star (max-change) and Z_last
# ---------------------------------------------------------------------------

@dataclass
class TickSnapshots:
    z_star: Optional[Tensor]
    z_last: Optional[Tensor]
    t_star: Optional[int]
    t_last: Optional[int]


class TickRepresentativeCollector:
    """Select representative internal ticks without storing all ticks.

    We want a label-free, task-agnostic snapshot that captures *dynamics*.
    A simple and robust proxy is:
      Z_star := state at tick with maximal step-to-step change (mean L2 over batch)
      Z_last := state at final tick

    This requires only streaming comparison with the previous tick.
    """
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
# 3) Streaming standardization + reservoir sampling
# ---------------------------------------------------------------------------

def _rms_normalize_rows(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise RMS normalization: z / sqrt(mean(z^2)+eps)."""
    rms = np.sqrt(np.mean(z * z, axis=1, keepdims=True) + eps)
    return z / rms


class Reservoir:
    """Uniform reservoir sampler over rows, with streaming standardization via sklearn.

    Mental model:
      We need a stable low-rank Gram proxy. That requires:
        (a) many activation samples,
        (b) robust standardization,
        (c) bounded memory.

      We therefore:
        - RMS-normalize each row to damp scale drift
        - update a StandardScaler online (mean/var over all seen rows)
        - retain a uniform reservoir sample of fixed size for PCA/SVD

    This gives a dataset-agnostic activation matrix X without storing everything.
    """
    def __init__(self, d: int, capacity: int, clip: float = 5.0):
        self.d = int(d)
        self.capacity = int(capacity)
        self.clip = float(clip)

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.n_seen = 0
        self.size = 0
        self.data = np.empty((self.capacity, self.d), dtype=np.float32)

    def add_batch(self, z_torch: Tensor):
        if z_torch is None or z_torch.numel() == 0:
            return
        z = z_torch.detach().to(device="cpu", dtype=torch.float32).numpy()
        z = _rms_normalize_rows(z)  # (B,D)
        self.scaler.partial_fit(z)

        b = z.shape[0]
        # fill
        if self.size < self.capacity:
            n_fill = min(self.capacity - self.size, b)
            self.data[self.size:self.size + n_fill] = z[:n_fill]
            self.size += n_fill
            self.n_seen += n_fill
            z = z[n_fill:]
        if z.size == 0:
            return

        # replacement (vectorized reservoir)
        br = z.shape[0]
        # indices sampled uniformly in [0, n_seen + t]
        n_seen_t = self.n_seen + np.arange(1, br + 1, dtype=np.int64)
        j = (np.random.rand(br) * n_seen_t.astype(np.float64)).astype(np.int64)
        mask = j < self.capacity
        if np.any(mask):
            self.data[j[mask]] = z[mask]
        self.n_seen += br

    def finalize_X(self) -> np.ndarray:
        """Return standardized reservoir matrix X in float32."""
        if self.size < 2:
            return np.empty((0, self.d), dtype=np.float32)
        mu = self.scaler.mean_.astype(np.float32, copy=False)
        std = np.sqrt(self.scaler.var_.astype(np.float32, copy=False) + 1e-12)
        X = (self.data[:self.size] - mu[None, :]) / (std[None, :] + 1e-8)
        if self.clip is not None:
            X = np.clip(X, -self.clip, self.clip)
        return X.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 4) Low-rank Gram proxy + adaptive bottleneck allocation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LowRank:
    V: np.ndarray      # (D,k_eff)
    S: np.ndarray      # (k_eff,)
    p: np.ndarray      # (D,) leverage-tempered importance
    eta: float         # explained energy proxy
    k_eff: int


def lowrank_from_X(X: np.ndarray, d: int, k: int) -> LowRank:
    """Compute low-rank factors + a neuron importance distribution.

    We use randomized SVD for robustness and speed on CPU.
    From X (N,D), we compute top k singular values/vectors:
      X ≈ U diag(S) V^T

    Neuron "leverage" (importance) is:
      lev_i = ||V[i,:]||^2

    We temper lev toward uniform by how much energy the top-k captures:
      eta = sum(S^2) / ||X||_F^2
      alpha = 1 - eta
      p = (1-alpha)*lev_norm + alpha*uniform
    """
    if X.size == 0 or X.shape[0] < 2:
        p = np.full((d,), 1.0 / d, dtype=np.float32)
        return LowRank(V=np.zeros((d, 0), np.float32), S=np.zeros((0,), np.float32), p=p, eta=0.0, k_eff=0)

    Xc = X - X.mean(axis=0, keepdims=True)  # safe even if already standardized
    N = Xc.shape[0]
    k_eff = int(min(k, d, max(1, min(N - 1, k))))

    # randomized_svd returns Vt: (k_eff,D)
    _, S, Vt = randomized_svd(Xc, n_components=k_eff, n_iter=2, random_state=None)
    V = Vt.T.astype(np.float32, copy=False)
    S = S.astype(np.float32, copy=False)

    lev = np.sum(V * V, axis=1).astype(np.float64)
    lev_sum = float(np.sum(lev))
    if lev_sum <= 0:
        lev = np.full((d,), 1.0 / d, dtype=np.float64)
    else:
        lev = lev / lev_sum

    fro2 = float(np.sum(Xc * Xc))
    top2 = float(np.sum(S * S))
    eta = float(top2 / (fro2 + 1e-8))
    alpha = float(np.clip(1.0 - eta, 0.0, 1.0))

    p = (1.0 - alpha) * lev + alpha * (1.0 / d)
    p = np.maximum(p, 1e-12)
    p = (p / np.sum(p)).astype(np.float32, copy=False)

    return LowRank(V=V, S=S, p=p, eta=eta, k_eff=k_eff)


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
    """Compute |K| on a candidate set from low-rank factors.

    If K ≈ V diag(S^2) V^T, then on candidates C:
      K_C = (V_C diag(S)) (V_C diag(S))^T
    """
    A = (V[idx, :] * S[None, :]).astype(np.float32, copy=False)  # (M,k)
    G = np.abs(A @ A.T).astype(np.float32, copy=False)          # (M,M)
    np.fill_diagonal(G, 0.0)
    return G


def participation_ratio(q: np.ndarray) -> float:
    """Effective support size: 1 / sum(q^2)."""
    q = np.asarray(q, dtype=np.float64)
    q = np.maximum(q, 1e-12)
    q = q / np.sum(q)
    return float(1.0 / np.sum(q * q))


# ---------------------------------------------------------------------------
# 5) Deterministic greedy edge selection with degree caps
# ---------------------------------------------------------------------------

def greedy_degree_capped_edges(
    W: np.ndarray,
    J: int,
    d_max: int,
    forbid_self: bool = True,
) -> List[Tuple[int, int]]:
    """Pick up to J edges from W (square, nonnegative) via greedy score order under degree cap.

    This is the minimal “workhorse” primitive:
      - score is W_ij
      - accept highest score edges while degrees <= d_max
      - deterministic given W and tie ordering from argsort

    Complexity is fine for M<=1024 once at init time.
    """
    M = W.shape[0]
    if M <= 1 or J <= 0:
        return []

    tri_i, tri_j = np.triu_indices(M, k=1 if forbid_self else 0)
    scores = W[tri_i, tri_j]

    order = np.argsort(scores)[::-1]
    deg = np.zeros((M,), dtype=np.int32)
    out: List[Tuple[int, int]] = []

    for k in order:
        if len(out) >= J:
            break
        i = int(tri_i[k])
        j = int(tri_j[k])
        if forbid_self and i == j:
            continue
        if deg[i] >= d_max or deg[j] >= d_max:
            continue
        if scores[k] <= 0:
            break
        out.append((i, j))
        deg[i] += 1
        deg[j] += 1

    return out


def pairs_from_lowrank_adaptive(
    factors: LowRank,
    J: int,
    n_self: int,
    d: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Adaptive universal pairing policy from low-rank kernel proxy.

    Mental model:
      - Build a small candidate Gram proxy |K_C| of size MxM
      - Measure hubness distribution q over candidates from row-sums of |K_C|
      - Infer whether the kernel is bottleneck-friendly via n_eff(q)
      - Allocate edges: J_core (bottlenecked within top hubs) + J_wide (coverage)
      - Select edges deterministically by greedy degree-capped scores

    This yields a universal default:
      - If kernel has strong hubs: behaves dense-like (helps tasks like mazes)
      - If kernel diffuse: behaves coverage-like (helps perceptual tasks)
      - No task flags; no tuning knobs.
    """
    J = int(J)
    d = int(d)
    n_self = int(min(n_self, max(0, J - 1)))

    # fallback for missing factors
    if factors.k_eff <= 0:
        left = np.random.randint(0, d, size=J, dtype=np.int64)
        right = np.random.randint(0, d, size=J, dtype=np.int64)
        # enforce exact self-prefix if requested
        if n_self > 0:
            top = np.argsort(factors.p)[::-1][:n_self]
            left[:n_self] = top
            right[:n_self] = top
        return left, right, {"mode": "uniform_fallback"}

    # choose candidate set C
    C = sample_without_replacement(factors.p, M)  # indices in [0..D)
    G = gram_abs_from_factors(factors.V[:, :factors.k_eff], factors.S[:factors.k_eff], C)  # (M,M)

    # hubness q over candidates
    s = np.sum(G, axis=1).astype(np.float64)
    s = np.maximum(s, 1e-12)
    q = (s / np.sum(s)).astype(np.float32, copy=False)
    n_eff = participation_ratio(q)

    # Non-self edges are what the graph sampler actually selects (self edges are forced prefix).
    J_nonself = J - n_self
    n_tri = n_triangle(J_nonself)  # node-count implied by the non-self edge budget
    f = float(min(1.0, n_tri / max(1e-6, n_eff)))  # bottleneck fraction inferred from kernel

    J_core = int(round(f * J_nonself))
    J_wide = J_nonself - J_core

    # build scores on candidates: W_ij = |K_ij| * sqrt(q_i q_j)
    sqrt_q = np.sqrt(q.astype(np.float64) + 1e-12)
    W = (G * (sqrt_q[:, None] * sqrt_q[None, :]).astype(np.float32)).astype(np.float32, copy=False)
    np.fill_diagonal(W, 0.0)

    # self-pairs: choose top candidates by q, mapped back to global indices
    left_pairs: List[int] = []
    right_pairs: List[int] = []
    used = set()

    if n_self > 0:
        top_local = np.argsort(q)[::-1][:n_self]
        top_global = C[top_local]
        for idx in top_global.tolist():
            left_pairs.append(int(idx))
            right_pairs.append(int(idx))
            used.add((int(idx), int(idx)))

    # Size the core candidate pool based on the *core* edge budget, not total.
    # Otherwise, when f is small you dilute the "core" and lose the intended dense-like behavior.
    Bn = int(min(M, max(2, n_triangle(J_core)))) if J_core > 0 else 0
    B = np.argsort(q)[::-1][:Bn] if Bn > 0 else np.asarray([], dtype=np.int64)

    def _degree_cap_for(J_edges: int, n_nodes_available: int) -> int:
        """Degree cap mirroring the earlier triangular-n_target heuristic."""
        J_edges = int(max(0, J_edges))
        n_nodes_available = int(max(0, n_nodes_available))
        if J_edges <= 0 or n_nodes_available <= 1:
            return 0

        n_tri_local = n_triangle(J_edges)  # nodes needed to realize J_edges (non-self)
        n_target = n_tri_local if n_tri_local <= n_nodes_available else n_nodes_available
        n_target = max(1, n_target)

        return int(math.ceil(2.0 * J_edges / n_target))

    # select edges within B
    if J_core > 0 and Bn > 1:
        WB = W[np.ix_(B, B)]
        dmax_core = _degree_cap_for(J_core, Bn)
        edges_core = greedy_degree_capped_edges(WB, J_core, dmax_core, forbid_self=True)
        for (iB, jB) in edges_core:
            i = int(C[B[iB]])
            j = int(C[B[jB]])
            if (i, j) in used or i == j:
                continue
            used.add((i, j))
            left_pairs.append(i)
            right_pairs.append(j)

    # wide edges: full candidates with diffuse cap
    if J_wide > 0 and M > 1:
        dmax_wide = _degree_cap_for(J_wide, M)
        edges_wide = greedy_degree_capped_edges(W, J_wide, dmax_wide, forbid_self=True)
        for (iC, jC) in edges_wide:
            i = int(C[iC])
            j = int(C[jC])
            if (i, j) in used or i == j:
                continue
            used.add((i, j))
            left_pairs.append(i)
            right_pairs.append(j)

    # If we didn't reach J, fill remaining uniformly (guaranteed termination, no extra self pairs)
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

    left = np.asarray(left_pairs[:J], dtype=np.int64)
    right = np.asarray(right_pairs[:J], dtype=np.int64)

    diag = {
        "mode": "adaptive",
        "M": int(M),
        "n_tri": int(n_tri),
        "n_eff": float(n_eff),
        "f_bottleneck": float(f),
        "J_core": int(J_core),
        "J_wide": int(J_wide),
        "Bn": int(Bn),
        "self": int(n_self),
        "dmax_core": int(dmax_core) if "dmax_core" in locals() else -1,
        "dmax_wide": int(dmax_wide) if "dmax_wide" in locals() else -1,
    }
    return left, right, diag


# ---------------------------------------------------------------------------
# 6) Task-agnostic initialization entrypoint
# ---------------------------------------------------------------------------

def default_warmup_batches(args_warmup_steps: int) -> int:
    """Occam default: warmup is about statistics, not LR scheduling."""
    if args_warmup_steps and args_warmup_steps > 0:
        return int(min(1024, args_warmup_steps))
    return 1024


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
    device: Optional[torch.device | str] = None,
    batch_to_x: Optional[Callable[[Any], Tensor]] = None,
    preserve_seed: bool = True,
) -> Dict[str, Any]:
    """One-shot pairing initialization for CTM (action/out), task-agnostic.

    Expected model interface:
      - attributes: d_model, n_synch_action, n_synch_out
      - methods:
          resample_action_pairs_(n_random_pairing_self: int)  [optional but recommended]
          set_neuron_pairs(role: str, left: Tensor, right: Tensor)
      - forward signature: model(x, track=False, callback=collector)

    High-level algorithm:
      1) Warmup: gather Z_star (and Z_last) states from real data forwards
      2) Build standardized reservoir matrices X_action, X_out
      3) Estimate low-rank Gram proxy factors for each role
      4) Sample pairs using adaptive bottleneck allocation from kernel hubness
      5) Write indices into model buffers once
    """
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

    # design constants
    k = int(min(256, d))
    n_res = int(min(32768, 32 * k))  # bounded memory
    M = int(min(d, 4 * k))           # bounded quadratic object

    # reservoirs (role-specific)
    res_action = Reservoir(d=d, capacity=n_res, clip=5.0)
    res_out = Reservoir(d=d, capacity=n_res, clip=5.0)

    collector = TickRepresentativeCollector()
    tstars: List[int] = []

    with preserve_rng_state(preserve_seed), batchnorm_use_batch_stats_no_update(model), torch.inference_mode():
        it = iter(dataloader)
        for _ in range(int(warmup_batches)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)

            x = batch_to_x(batch).to(device, non_blocking=True)

            # Break circularity: resample action pairs only (action affects trajectories)
            if hasattr(model, "resample_action_pairs_"):
                model.resample_action_pairs_(int(n_random_pairing_self))

            collector.reset()
            _ = model(x, track=False, callback=collector)
            snaps = collector.snapshots()
            if snaps.t_star is not None:
                tstars.append(int(snaps.t_star))
            if snaps.z_star is None or snaps.z_last is None:
                continue

            # action: Z_star only
            res_action.add_batch(snaps.z_star)
            # out: Z_star + Z_last
            res_out.add_batch(snaps.z_star)
            res_out.add_batch(snaps.z_last)

    X_action = res_action.finalize_X()
    X_out = res_out.finalize_X()

    factors_action = lowrank_from_X(X_action, d=d, k=k)
    factors_out = lowrank_from_X(X_out, d=d, k=k)

    left_a, right_a, diag_a = pairs_from_lowrank_adaptive(
        factors_action, J=J_action, n_self=n_random_pairing_self, d=d, M=M
    )
    left_o, right_o, diag_o = pairs_from_lowrank_adaptive(
        factors_out, J=J_out, n_self=n_random_pairing_self, d=d, M=M
    )

    model.set_neuron_pairs("action", torch.from_numpy(left_a).to(device=device), torch.from_numpy(right_a).to(device=device))
    model.set_neuron_pairs("out", torch.from_numpy(left_o).to(device=device), torch.from_numpy(right_o).to(device=device))

    t_arr = np.asarray(tstars, dtype=np.float32) if tstars else np.asarray([0.0], dtype=np.float32)
    diag = {
        "warmup_batches": int(warmup_batches),
        "reservoir_capacity": int(n_res),
        "reservoir_size_action": int(res_action.size),
        "reservoir_seen_action": int(res_action.n_seen),
        "reservoir_size_out": int(res_out.size),
        "reservoir_seen_out": int(res_out.n_seen),
        "t_star_mean": float(np.mean(t_arr)),
        "t_star_std": float(np.std(t_arr)),
        "action": {
            "J": int(J_action),
            "k_eff": int(factors_action.k_eff),
            "eta": float(factors_action.eta),
            **diag_a,
        },
        "out": {
            "J": int(J_out),
            "k_eff": int(factors_out.k_eff),
            "eta": float(factors_out.eta),
            **diag_o,
        },
    }
    return diag
