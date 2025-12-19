from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Iterable, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


Tensor = torch.Tensor


@dataclass
class PairRoleDiagnostics:
    role: str
    J: int
    n_self: int
    unique_pair_frac: float
    self_pair_frac: float
    unique_neurons: int
    deg_min: int
    deg_median: float
    deg_max: int
    eta: float
    alpha: float
    k_eff: int
    p_entropy_norm: float
    d_max_final: int
    d_max_increments: int
    used_ultimate_fallback: bool


@dataclass
class WarmupDiagnostics:
    warmup_batches: int
    reservoir_capacity: int
    reservoir_size_action: int
    reservoir_size_out: int
    reservoir_seen_action: int
    reservoir_seen_out: int
    t_star_mean: float
    t_star_std: float
    t_star_extreme_frac: float  # fraction of batches with t_star in {0, T-1}


@contextlib.contextmanager
def preserve_rng_state(enabled: bool = True):
    """Preserve and restore RNG states for torch / numpy / python."""
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
    """Warmup mode: Dropout OFF, BN uses batch stats, BN buffers unchanged.

    - Forces model.eval() (disables Dropout).
    - Forces BN layers into train mode so they use batch statistics.
    - Prevents BN running_mean/var updates by setting momentum=0.0.
    - Snapshots/restores BN buffers + training flags.
    - Restores the model's original train/eval state.
    """
    saved_bn: List[Tuple[nn.Module, Dict[str, Any]]] = []
    model_training = model.training
    try:
        # Disable dropout and other train-time stochasticity globally.
        model.eval()

        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                st: Dict[str, Any] = {
                    "training": m.training,
                    "momentum": m.momentum,
                }
                if getattr(m, "running_mean", None) is not None:
                    st["running_mean"] = m.running_mean.detach().clone()
                if getattr(m, "running_var", None) is not None:
                    st["running_var"] = m.running_var.detach().clone()
                if getattr(m, "num_batches_tracked", None) is not None:
                    st["num_batches_tracked"] = m.num_batches_tracked.detach().clone()
                saved_bn.append((m, st))

                # Use batch stats.
                m.train(True)
                # Avoid drifting running stats.
                m.momentum = 0.0

        yield
    finally:
        # Restore BN module state.
        for m, st in saved_bn:
            m.train(st["training"])
            m.momentum = st["momentum"]
            if "running_mean" in st:
                m.running_mean.copy_(st["running_mean"])
            if "running_var" in st:
                m.running_var.copy_(st["running_var"])
            if "num_batches_tracked" in st:
                m.num_batches_tracked.copy_(st["num_batches_tracked"])

        # Restore overall model mode.
        model.train(model_training)


@dataclass
class TickSnapshots:
    z_star: Optional[Tensor] = None
    z_last: Optional[Tensor] = None


class TickRepresentativeCollector:
    """Callback collector to select Z_star (max change) and Z_last (final tick) per forward pass.

    Assumes callback is invoked once per internal tick with (tick_idx, activated_state),
    where activated_state is the post-update state for that tick.
    """
    def __init__(self):
        self.prev: Optional[Tensor] = None
        self.best_delta: float = -1.0
        self.best_z: Optional[Tensor] = None
        self.last_z: Optional[Tensor] = None
        self.best_tick: Optional[int] = None
        self.last_tick: Optional[int] = None

    def __call__(self, tick_idx: int, activated_state: Tensor):
        # activated_state: (B, D)
        z = activated_state.detach()
        self.last_z = z
        self.last_tick = tick_idx
        if self.prev is not None:
            # Δ_t = mean over batch of L2 norm of step difference
            delta = (z - self.prev).norm(dim=1).mean().item()
            if delta > self.best_delta:
                self.best_delta = delta
                self.best_z = z
                self.best_tick = tick_idx
        self.prev = z

    def snapshots(self) -> TickSnapshots:
        # If best_z was never set (e.g., only one tick), fall back to last.
        z_star = self.best_z if self.best_z is not None else self.last_z
        return TickSnapshots(z_star=z_star, z_last=self.last_z)

    def reset(self):
        self.prev = None
        self.best_delta = -1.0
        self.best_z = None
        self.last_z = None
        self.best_tick = None
        self.last_tick = None

    def t_star(self) -> Optional[int]:
        return self.best_tick if self.best_tick is not None else self.last_tick


class WelfordStats:
    """Streaming mean/variance (per-dimension) using Welford, with batch combine."""
    def __init__(self, d: int, device: torch.device):
        self.count = 0
        self.mean = torch.zeros(d, device=device, dtype=torch.float64)
        self.M2 = torch.zeros(d, device=device, dtype=torch.float64)

    def update_batch(self, X: Tensor):
        """Update from X: (B, D) float32 on self.mean.device."""
        if X.numel() == 0:
            return
        X64 = X.to(dtype=torch.float64)
        b = X64.shape[0]

        batch_mean = X64.mean(dim=0)
        batch_M2 = ((X64 - batch_mean) ** 2).sum(dim=0)

        if self.count == 0:
            self.count = b
            self.mean.copy_(batch_mean)
            self.M2.copy_(batch_M2)
            return

        n = self.count
        n_new = n + b
        delta = batch_mean - self.mean
        self.mean += delta * (b / n_new)
        self.M2 += batch_M2 + (delta * delta) * (n * b / n_new)
        self.count = n_new

    def finalize(self) -> Tuple[Tensor, Tensor]:
        if self.count < 2:
            var = torch.ones_like(self.mean)
        else:
            var = self.M2 / (self.count - 1)
        std = torch.sqrt(var.clamp_min(1e-12))
        return self.mean.to(dtype=torch.float32), std.to(dtype=torch.float32)



class Reservoir:
    """Uniform reservoir sampler for rows in R^D, plus running stats over all seen rows."""
    def __init__(self, d: int, capacity: int, device: torch.device):
        self.d = d
        self.capacity = int(capacity)
        self.device = device  # where reservoir + stats live (typically CPU)
        self.stats = WelfordStats(d=d, device=device)
        self.n_seen = 0
        self.data = torch.empty((self.capacity, d), device=device, dtype=torch.float32)
        self.size = 0

    @staticmethod
    def rms_normalize_batch(Z: Tensor, eps: float = 1e-8) -> Tensor:
        # Z: (B, D) float32
        rms = torch.sqrt(Z.pow(2).mean(dim=1, keepdim=True) + eps)
        return Z / rms

    def add_batch(self, Z: Tensor):
        # Z: (B, D) on any device
        Z_cpu = Z.detach().to(device=self.device, dtype=torch.float32)
        if Z_cpu.numel() == 0:
            return

        Zn = self.rms_normalize_batch(Z_cpu)  # (B, D)
        self.stats.update_batch(Zn)

        B = Zn.shape[0]

        # Fill phase
        if self.size < self.capacity:
            n_fill = min(self.capacity - self.size, B)
            self.data[self.size:self.size + n_fill].copy_(Zn[:n_fill])
            self.size += n_fill
            self.n_seen += n_fill
            Z_remaining = Zn[n_fill:]
        else:
            Z_remaining = Zn

        if Z_remaining.numel() == 0:
            return

        # Replacement phase (vectorized reservoir sampling)
        Br = Z_remaining.shape[0]
        # For each incoming row t (0..Br-1), sample j_t ~ Unif{0..(n_seen+t)}
        # Use torch RNG (preserved by preserve_rng_state).
        offsets = torch.arange(1, Br + 1, device=self.device, dtype=torch.int64)
        n_seen_t = self.n_seen + offsets  # (Br,)
        u = torch.rand(Br, device=self.device)
        j = torch.floor(u * n_seen_t.to(dtype=torch.float32)).to(dtype=torch.int64)  # (Br,)
        mask = j < self.capacity
        if mask.any():
            self.data[j[mask]].copy_(Z_remaining[mask])

        self.n_seen += Br

    def finalize_matrix(self, clip: Optional[float] = 5.0, eps: float = 1e-8) -> Tensor:
        """Return standardized reservoir matrix X in float32 on self.device."""
        mu, std = self.stats.finalize()
        X = (self.data[:self.size] - mu) / (std + eps)
        if clip is not None:
            X = X.clamp(min=-clip, max=clip)
        return X


@dataclass
class LowRankFactors:
    V: Tensor   # (D, k_eff)
    S: Tensor   # (k_eff,)
    p: Tensor   # (D,)
    eta: float
    k_eff: int


def lowrank_leverage_distribution(
    X: Tensor, d: int, k: int, device: torch.device, eps: float = 1e-8
) -> LowRankFactors:
    """Compute tempered leverage distribution p over D neurons from reservoir matrix X.

    Robust to small N: uses q <= min(N, D) and k_eff <= q.
    """
    if X.numel() == 0 or X.shape[0] < 2:
        p = torch.full((d,), 1.0 / d, dtype=torch.float32, device=device)
        V = torch.zeros((d, 0), dtype=torch.float32, device=device)
        S = torch.zeros((0,), dtype=torch.float32, device=device)
        return LowRankFactors(V=V, S=S, p=p, eta=0.0, k_eff=0)

    Xd = X.to(device=device, dtype=torch.float32)

    # Center (safe even if X already standardized)
    Xd = Xd - Xd.mean(dim=0, keepdim=True)

    N = Xd.shape[0]
    q = min(k + 8, d, N)       # must satisfy q <= min(N, D)
    k_eff = min(k, q)

    if q <= 0 or k_eff <= 0:
        p = torch.full((d,), 1.0 / d, dtype=torch.float32, device=device)
        V = torch.zeros((d, 0), dtype=torch.float32, device=device)
        S = torch.zeros((0,), dtype=torch.float32, device=device)
        return LowRankFactors(V=V, S=S, p=p, eta=0.0, k_eff=0)

    # torch.pca_lowrank returns V: (D, q)
    _, S, V = torch.pca_lowrank(Xd, q=q, center=False)
    V = V[:, :k_eff].contiguous()
    S = S[:k_eff].contiguous()

    # leverage over neurons
    lev = V.pow(2).sum(dim=1).clamp_min(0.0)
    lev_sum = lev.sum()
    lev = (lev / lev_sum) if lev_sum.item() > 0 else torch.full_like(lev, 1.0 / d)

    fro2 = float(Xd.pow(2).sum().item())
    top2 = float(S.pow(2).sum().item())
    eta = float(top2 / (fro2 + eps))
    alpha = float(max(0.0, min(1.0, 1.0 - eta)))

    p = (1.0 - alpha) * lev + alpha * (1.0 / d)
    p = (p + 1e-12) / p.sum()

    return LowRankFactors(V=V, S=S, p=p.detach(), eta=eta, k_eff=k_eff)


def _sample_without_replacement(p: Tensor, m: int, device: torch.device) -> Tensor:
    p = p.to(device=device, dtype=torch.float32)
    p = (p + 1e-12) / p.sum()
    m = min(int(m), p.numel())
    return torch.multinomial(p, m, replacement=False)


def _build_candidate_gram(V: Tensor, S: Tensor, candidates: Tensor) -> Tensor:
    A = V[candidates] * S.unsqueeze(0)  # (M, k_eff)
    # R_ij = <A_i, A_j>
    return torch.einsum("ik,jk->ij", A, A)


def sample_pairs_from_factors(
    factors: LowRankFactors,
    J: int,
    n_self: int,
    d: int,
    device: torch.device,
    L: int = 64,
    max_attempts: int = 64,
    global_attempts_mult: int = 10,
    return_info: bool = False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Sample J pairs (left,right) using degree-capped, correlation-guided sampling with termination guarantees."""
    J = int(J)
    n_self = int(min(n_self, J))
    p = factors.p.to(device=device)
    V = factors.V.to(device=device)
    S = factors.S.to(device=device)

    k_eff = factors.k_eff
    if k_eff <= 0:
        # Pure uniform fallback (still respect self-pair prefix)
        left = torch.randint(0, d, (J,), device=device)
        right = torch.randint(0, d, (J,), device=device)
        if n_self > 0:
            top_self = torch.topk(factors.p.to(device=device), k=n_self, largest=True).indices
            left[:n_self] = top_self
            right[:n_self] = top_self
        return left.long(), right.long()

    M = min(d, 4 * k_eff)
    candidates = _sample_without_replacement(p, M, device=device)  # (M,)
    pC = p[candidates]
    pC = (pC + 1e-12) / pC.sum()
    sqrt_pC = torch.sqrt(pC)

    R = _build_candidate_gram(V, S, candidates)  # (M, M)
    absR = R.abs()
    absR.fill_diagonal_(0.0)

    L_eff = min(int(L), M - 1) if M > 1 else 0
    # Precompute neighbor indices for each candidate row
    neigh_idx = []
    neigh_w = []
    if L_eff > 0:
        topk_vals, topk_idx = torch.topk(absR, k=L_eff, dim=1, largest=True, sorted=False)
        neigh_idx = topk_idx  # (M, L)
        # weights will be computed per i using p_j
        neigh_w = topk_vals  # (M, L)

    left_pairs: List[int] = []
    right_pairs: List[int] = []
    used = set()
    deg = torch.zeros(M, device=device, dtype=torch.int32)

    # Add self-pairs first: pick top p globally, deterministic
    if n_self > 0:
        top_self = torch.topk(p, k=n_self, largest=True).indices
        for idx in top_self.tolist():
            left_pairs.append(int(idx))
            right_pairs.append(int(idx))
            used.add((int(idx), int(idx)))

    # Degree cap over candidates (for non-self pairs)
    d_max = int(math.ceil(2 * max(1, J) / max(1, M)))
    d_max_start = d_max
    d_max_increments = 0
    used_ultimate_fallback = False

    total_attempts = 0
    target_nonself = J - n_self

    def _unsat_candidates_mask():
        return (deg < d_max)

    def _choose_uniform_unsat(exclude_i: Optional[int] = None) -> Optional[int]:
        mask = _unsat_candidates_mask()
        if exclude_i is not None:
            mask[exclude_i] = False
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        if idxs.numel() == 0:
            return None
        j = idxs[torch.randint(0, idxs.numel(), (1,), device=device)].item()
        return int(j)

    # Helper to record an edge
    def _accept(i_c: int, j_c: int) -> bool:
        # i_c, j_c are candidate indices [0..M-1]
        if i_c == j_c:
            return False
        if deg[i_c] >= d_max or deg[j_c] >= d_max:
            return False
        i_g = int(candidates[i_c].item())
        j_g = int(candidates[j_c].item())
        key = (i_g, j_g)
        if key in used:
            return False
        used.add(key)
        left_pairs.append(i_g)
        right_pairs.append(j_g)
        deg[i_c] += 1
        deg[j_c] += 1
        return True

    # Main loop
    while len(left_pairs) - n_self < target_nonself:
        if total_attempts > global_attempts_mult * max(1, target_nonself):
            # Ultimate fallback: fill remaining with uniform random pairs over [0, d)
            # Fill until we have target_total pairs; guaranteed termination.
            while len(left_pairs) < (n_self + target_nonself):
                i_g = int(torch.randint(0, d, (1,), device=device).item())
                j_g = int(torch.randint(0, d, (1,), device=device).item())
                # Avoid creating additional self-pairs beyond the explicit n_self prefix.
                if d > 1 and j_g == i_g:
                    j_g = (i_g + 1) % d
                if (i_g, j_g) in used:
                    continue
                used.add((i_g, j_g))
                left_pairs.append(i_g)
                right_pairs.append(j_g)
            used_ultimate_fallback = True
            break

        # sample left endpoint among candidates
        i_c = int(torch.multinomial(pC, 1, replacement=True).item())
        attempts = 0
        accepted = False
        while attempts < max_attempts and not accepted:
            total_attempts += 1
            attempts += 1

            if deg[i_c] >= d_max:
                # choose a new i
                i_c = int(torch.multinomial(pC, 1, replacement=True).item())
                continue

            if L_eff > 0:
                neighs = neigh_idx[i_c]  # (L,)
                # compute weights w_ij = sqrt(p_i p_j) * |R_ij|
                # p_i via sqrt_pC[i_c], p_j via sqrt_pC[neigh]
                w = sqrt_pC[i_c] * sqrt_pC[neighs] * neigh_w[i_c]
                w_sum = w.sum()
                if w_sum.item() <= 0:
                    # fallback to uniform among neighbors
                    j_c = int(neighs[torch.randint(0, neighs.numel(), (1,), device=device)].item())
                else:
                    w = (w + 1e-12) / w_sum
                    j_c = int(neighs[torch.multinomial(w, 1, replacement=True)].item())
            else:
                # no neighbors; fall back immediately
                j_c = _choose_uniform_unsat(exclude_i=i_c)
                if j_c is None:
                    accepted = False
                    break

            # Try accept
            accepted = _accept(i_c, j_c)

        if accepted:
            continue

        # Staged fallback if couldn't accept within max_attempts:
        # 1) keep degree cap; choose uniform unsaturated j
        j_c = _choose_uniform_unsat(exclude_i=i_c)
        if j_c is not None and _accept(i_c, j_c):
            continue

        # 2) relax degree cap minimally
        d_max += 1
        d_max_increments += 1
        # try once more with relaxed cap
        j_c = _choose_uniform_unsat(exclude_i=i_c)
        if j_c is not None and _accept(i_c, j_c):
            continue

    left = torch.tensor(left_pairs[:J], device=device, dtype=torch.long)
    right = torch.tensor(right_pairs[:J], device=device, dtype=torch.long)

    if return_info:
        info = {
            "d_max_start": d_max_start,
            "d_max_final": d_max,
            "d_max_increments": d_max_increments,
            "used_ultimate_fallback": used_ultimate_fallback,
        }
        return left, right, info

    return left, right


def default_warmup_batches(args_warmup_steps: int) -> int:
    """Occam + safe default: cap warmup batches to avoid surprising slowdowns."""
    if args_warmup_steps and args_warmup_steps > 0:
        return int(min(1024, args_warmup_steps))
    return 1024


def _default_batch_to_x(batch: Any) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        raise ValueError("Batch is a dict; please provide batch_to_x.")
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def initialize_ctm_pairs(
    model: nn.Module,
    dataloader: Iterable,
    warmup_batches: int,
    n_random_pairing_self: int,
    device: str | torch.device | None = None,
    batch_to_x: Optional[Callable[[Any], Tensor]] = None,
    seed_preserve: bool = True,
) -> None:
    """One-shot initialization of CTM action/out random-pairing indices using warmup batches.

    Requirements:
    - `model` must expose:
        - `.d_model`, `.n_synch_action`, `.n_synch_out`
        - `.resample_action_pairs_(n_random_pairing_self)`
        - `.set_neuron_pairs(synch_type, left, right)`
        - `.forward(..., callback=...)`

    This function is task-agnostic: it only needs a dataloader and a way to extract x.
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
    k = min(256, d)
    n_res = min(32768, 32 * k)  # memory bound

    # Reservoirs on CPU for stability/memory
    cpu = torch.device("cpu")
    res_action = Reservoir(d=d, capacity=n_res, device=cpu)
    res_out = Reservoir(d=d, capacity=n_res, device=cpu)

    collector = TickRepresentativeCollector()

    # Warmup pass
    tstars: List[int] = []
    with preserve_rng_state(seed_preserve), batchnorm_use_batch_stats_no_update(model), torch.inference_mode():
        it = iter(dataloader)
        for _ in tqdm(range(int(warmup_batches)), desc="Synchronization Init Warmup"):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)

            x = batch_to_x(batch).to(device, non_blocking=True)

            # Break circularity: resample action pairs only
            if hasattr(model, "resample_action_pairs_"):
                model.resample_action_pairs_(n_random_pairing_self=int(n_random_pairing_self))

            collector.reset()
            _ = model(x, track=False, callback=collector)  # forward only

            snaps = collector.snapshots()
            t_star = collector.t_star()
            if t_star is not None:
                tstars.append(int(t_star))
            if snaps.z_star is None or snaps.z_last is None:
                continue

            # Action statistics: Z_star only
            res_action.add_batch(snaps.z_star)
            # Out statistics: Z_star and Z_last
            res_out.add_batch(snaps.z_star)
            res_out.add_batch(snaps.z_last)

    # Finalize and compute factors/pairs (move compute to target device)
    X_action = res_action.finalize_matrix(clip=5.0)
    X_out = res_out.finalize_matrix(clip=5.0)

    factors_action = lowrank_leverage_distribution(X_action, d=d, k=k, device=device)
    factors_out = lowrank_leverage_distribution(X_out, d=d, k=k, device=device)

    # L depends on M which depends on k_eff
    L_a = min(64, max(0, min(d, 4 * factors_action.k_eff) - 1))
    L_o = min(64, max(0, min(d, 4 * factors_out.k_eff) - 1))

    left_a, right_a, info_a = sample_pairs_from_factors(
        factors=factors_action,
        J=J_action,
        n_self=n_random_pairing_self,
        d=d,
        device=device,
        L=L_a,
        return_info=True,
    )
    left_o, right_o, info_o = sample_pairs_from_factors(
        factors=factors_out,
        J=J_out,
        n_self=n_random_pairing_self,
        d=d,
        device=device,
        L=L_o,
        return_info=True,
    )

    # Write into model buffers once
    model.set_neuron_pairs("action", left_a, right_a)
    model.set_neuron_pairs("out", left_o, right_o)

    t_arr = np.array(tstars, dtype=np.float32) if len(tstars) else np.array([0.0], dtype=np.float32)
    t_mean = float(t_arr.mean())
    t_std = float(t_arr.std())
    t_extreme = float(((t_arr == t_arr.min()) | (t_arr == t_arr.max())).mean()) if len(tstars) else 0.0

    warm_diag = WarmupDiagnostics(
        warmup_batches=int(warmup_batches),
        reservoir_capacity=int(n_res),
        reservoir_size_action=int(res_action.size),
        reservoir_size_out=int(res_out.size),
        reservoir_seen_action=int(res_action.n_seen),
        reservoir_seen_out=int(res_out.n_seen),
        t_star_mean=t_mean,
        t_star_std=t_std,
        t_star_extreme_frac=t_extreme,
    )

    role_diag_a = _pair_role_diagnostics(
        role="action",
        left=left_a, right=right_a,
        J=J_action, n_self=n_random_pairing_self, d=d,
        factors=factors_action, sampler_info=info_a,
    )
    role_diag_o = _pair_role_diagnostics(
        role="out",
        left=left_o, right=right_o,
        J=J_out, n_self=n_random_pairing_self, d=d,
        factors=factors_out, sampler_info=info_o,
    )

    log_pair_init_diagnostics(warm_diag, [role_diag_a, role_diag_o], log_fn=print)


def _entropy_normalized(p: Tensor, eps: float = 1e-12) -> float:
    p = p.detach().float()
    p = (p + eps) / (p.sum() + eps)
    H = -(p * (p + eps).log()).sum().item()
    return float(H / math.log(p.numel() + 1e-12))


def _pair_role_diagnostics(
    role: str,
    left: Tensor,
    right: Tensor,
    J: int,
    n_self: int,
    d: int,
    factors: LowRankFactors,
    sampler_info: Dict[str, Any],
) -> PairRoleDiagnostics:
    left = left.detach().cpu()
    right = right.detach().cpu()

    pairs = torch.stack([left, right], dim=1)  # (J,2)
    unique_pairs = torch.unique(pairs, dim=0).shape[0]
    unique_pair_frac = float(unique_pairs / max(1, J))

    self_ct = int((left == right).sum().item())
    self_pair_frac = float(self_ct / max(1, J))

    touched = torch.unique(torch.cat([left, right], dim=0))
    unique_neurons = int(touched.numel())

    deg = torch.bincount(torch.cat([left, right]), minlength=d)  # degree counts
    deg_nonzero = deg[deg > 0]
    if deg_nonzero.numel() == 0:
        deg_min, deg_med, deg_max = 0, 0.0, 0
    else:
        deg_min = int(deg_nonzero.min().item())
        deg_max = int(deg_nonzero.max().item())
        deg_med = float(deg_nonzero.median().item())

    eta = float(factors.eta)
    alpha = float(max(0.0, min(1.0, 1.0 - eta)))
    p_ent = _entropy_normalized(factors.p.cpu())

    return PairRoleDiagnostics(
        role=role,
        J=int(J),
        n_self=int(min(n_self, J)),
        unique_pair_frac=unique_pair_frac,
        self_pair_frac=self_pair_frac,
        unique_neurons=unique_neurons,
        deg_min=deg_min,
        deg_median=deg_med,
        deg_max=deg_max,
        eta=eta,
        alpha=alpha,
        k_eff=int(factors.k_eff),
        p_entropy_norm=p_ent,
        d_max_final=int(sampler_info.get("d_max_final", -1)),
        d_max_increments=int(sampler_info.get("d_max_increments", 0)),
        used_ultimate_fallback=bool(sampler_info.get("used_ultimate_fallback", False)),
    )


def log_pair_init_diagnostics(warm: WarmupDiagnostics, roles: List[PairRoleDiagnostics], log_fn: Callable[[str], None] = print) -> None:
    log_fn("=== CTM Pair Initialization Diagnostics ===")
    log_fn(f"Warmup: batches={warm.warmup_batches}  reservoir_cap={warm.reservoir_capacity}")
    log_fn(f"Reservoir action: size={warm.reservoir_size_action} seen={warm.reservoir_seen_action}")
    log_fn(f"Reservoir out:    size={warm.reservoir_size_out} seen={warm.reservoir_seen_out}")
    log_fn(f"t_star: mean={warm.t_star_mean:.2f} std={warm.t_star_std:.2f} extreme_frac={warm.t_star_extreme_frac:.3f}")
    for r in roles:
        log_fn(f"[{r.role}] J={r.J} n_self={r.n_self} unique_pairs={r.unique_pair_frac:.3f} self_frac={r.self_pair_frac:.3f}")
        log_fn(f"[{r.role}] unique_neurons={r.unique_neurons} deg(min/med/max)={r.deg_min}/{r.deg_median:.1f}/{r.deg_max}")
        log_fn(f"[{r.role}] k_eff={r.k_eff} eta={r.eta:.3f} alpha={r.alpha:.3f} H(p)/logD={r.p_entropy_norm:.3f}")
        log_fn(f"[{r.role}] d_max_final={r.d_max_final} increments={r.d_max_increments} ultimate_fallback={r.used_ultimate_fallback}")


class SyncStatsCollector:
    """Collect per-batch sync health stats without storing full traces."""
    def __init__(self, near_zero_tol: float = 1e-3):
        self.near_zero_tol = near_zero_tol
        self._stds: Dict[str, float] = {}
        self._near_zero_frac: Dict[str, float] = {}
        self._last_tick: int = -1

    def __call__(self, tick_idx: int, role: str, sync_vec: Tensor):
        # sync_vec: (B, J)
        self._last_tick = tick_idx
        # Only keep last observed tick per role (cheap, and matches "health at current compute step")
        std = sync_vec.detach().float().std(dim=0)  # (J,)
        self._stds[role] = float(std.mean().item())
        self._near_zero_frac[role] = float((std < self.near_zero_tol).float().mean().item())

    def summary(self) -> Dict[str, Any]:
        return {
            "last_tick": self._last_tick,
            "action_sync_std_mean": self._stds.get("action", float("nan")),
            "out_sync_std_mean": self._stds.get("out", float("nan")),
            "action_sync_near_zero_frac": self._near_zero_frac.get("action", float("nan")),
            "out_sync_near_zero_frac": self._near_zero_frac.get("out", float("nan")),
        }


def summarize_max_certainty_tick(certainties: Any) -> Dict[str, float]:
    """Return mean/std of argmax certainty tick across batch."""
    if isinstance(certainties, list):
        cert = torch.stack([c.detach() for c in certainties], dim=1)  # (B,T)
    else:
        cert = certainties.detach()
        if cert.dim() == 2:
            pass
        elif cert.dim() == 3:
            # If model returns (T,B,...) style, try to coerce; keep conservative:
            cert = cert.squeeze(-1)
        else:
            return {"t2_mean": float("nan"), "t2_std": float("nan")}

    t2 = cert.argmax(dim=1).float()  # (B,)
    return {"t2_mean": float(t2.mean().item()), "t2_std": float(t2.std(unbiased=False).item())}