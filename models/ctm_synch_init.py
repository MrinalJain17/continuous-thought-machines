from __future__ import annotations

import contextlib
import dataclasses
import math
import os
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

Tensor = torch.Tensor
Pair = Tuple[int, int]


@dataclasses.dataclass(frozen=True)
class RoleBudgets:
    """
    Pair-feature budgets for CTM 'random-pairing' mode.

    CTM consumes a vector of synchronisation features where each feature is the product
    of two neuron activations for a chosen pair (i, j). Under 'random-pairing' these
    are explicit index buffers of length J for each role (out/action).

    Notes:
      - The initializer treats pairs as *unordered* (i<=j) for stability and interpretability.
      - CTM's compute_synchronisation uses left*right, so order does not matter.
    """
    J_out: int
    J_action: int
    n_self: int


@dataclasses.dataclass(frozen=True)
class WarmupConfig:
    """
    Warmup-only configuration: we collect traces on a fixed, neutral pairing and then freeze.

    warmup_batches:
      Number of batches used to build the sketch. In CTM each batch yields B*T observations
      because each internal tick contributes an activated_state vector.

    sampling:
      - "contiguous": take the next N batches from the iterator.
      - "spaced": sample N batches spread out over a larger prefix by skipping batches
                 (useful when dataloader order might be structured).
                 This does *not* involve training; it only changes which batches are used.

    target_observations:
      If warmup_batches is None, we choose it so that warmup_batches*B*T >= target_observations.
    """
    seed: int = 0
    preserve_global_rng_state: bool = True
    sampling: str = "contiguous"  # {"contiguous","spaced"}
    warmup_batches: Optional[int] = None
    target_observations: int = 100_000
    max_warmup_batches: int = 256


@dataclasses.dataclass(frozen=True)
class HeadRKDiagnostics:
    """
    Loggable summary for one-shot initialization.

    rho ~ 1  => low effective rank (hubby geometry)
    rho ~ 0  => high effective rank (diffuse geometry)
    """
    D: int
    k: int
    rho: float
    eff_rank: float
    n_obs: int
    rho_window_mean: float
    rho_window_std: float
    out: Dict[str, float]
    action: Dict[str, float]
    overlap_pairs: float
    overlap_nodes: float


@contextlib.contextmanager
def bn_batch_stats_no_update(model: nn.Module):
    """
    Warmup should not corrupt BN running stats. This context:
      - disables dropout via model.eval()
      - forces BN layers to use batch stats without updating running stats
    """
    BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    was_training = model.training
    saved = []
    model.eval()
    for m in model.modules():
        if isinstance(m, BN):
            st = (
                m, m.training, m.momentum,
                None if m.running_mean is None else m.running_mean.detach().clone(),
                None if m.running_var is None else m.running_var.detach().clone(),
                None if getattr(m, "num_batches_tracked", None) is None else m.num_batches_tracked.detach().clone(),
            )
            saved.append(st)
            m.train(True)
            m.momentum = 0.0
    try:
        yield
    finally:
        for m, tr, mom, rm, rv, nbt in saved:
            m.train(tr)
            m.momentum = mom
            if rm is not None: m.running_mean.copy_(rm)
            if rv is not None: m.running_var.copy_(rv)
            if nbt is not None: m.num_batches_tracked.copy_(nbt)
        model.train(was_training)


@contextlib.contextmanager
def neutral_action_pairs_for_warmup(model, *, n_self: int, seed: int):
    """
    Temporarily sets action pairs to baseline random-like for warmup and restores after.
    """
    left0 = model.action_neuron_indices_left.detach().clone()
    right0 = model.action_neuron_indices_right.detach().clone()

    left, right = baseline_random_action_pairs(model, n_self=n_self, seed=seed)
    model.set_neuron_pairs("action", left, right)
    try:
        yield
    finally:
        model.set_neuron_pairs("action", left0, right0)


def ctm_trace_fn(model, x: torch.Tensor) -> torch.Tensor:
    """
    Collect activated_state across internal ticks as a trace tensor (B, T, D).
    """
    states = []
    def cb(t: int, z: torch.Tensor):
        states.append(z)
    model(x, track=False, callback=cb, sync_callback=None)
    return torch.stack(states, dim=1)


def baseline_random_action_pairs(model, *, n_self: int, seed: int):
    """
    Deterministic neutral baseline for action pairs during warmup.

    Important: action pairing affects attention queries -> affects the dynamics you measure.
    Neutral baseline reduces circularity in a one-shot initializer.
    """
    device = model.action_neuron_indices_left.device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 999)

    D = int(model.d_model)
    J = int(model.n_synch_action)
    n_self = int(min(n_self, J))

    left = torch.randint(0, D, (J,), device=device, generator=g)
    right = torch.randint(0, D, (J,), device=device, generator=g)
    if n_self > 0:
        right[:n_self] = left[:n_self]
    return left, right


def initialize_headrk_pairs(
    *,
    model: nn.Module,
    dataloader: Iterable,
    trace_fn: Callable[[nn.Module, Tensor], Tensor],
    budgets: RoleBudgets,
    warmup: WarmupConfig,
    device: Optional[Union[str, torch.device]] = None,
    d_model: Optional[int] = None,
    batch_to_x: Callable = lambda b: b[0] if isinstance(b, (tuple, list)) else b,
    warmup_context: contextlib.AbstractContextManager = contextlib.nullcontext(),
    disjoint_roles: bool = False,
) -> Tuple[Dict[str, List[Pair]], Tensor, HeadRKDiagnostics]:
    """
    One-shot, warmup-only, kernel-aware CTM pair initialization (HEADR-K).

    The goal is a universal default that adapts across tasks without per-task knobs:
      - Warmup under a neutral baseline (random-like action pairing) outside this function.
      - Estimate trace geometry via a random projection sketch U = ZR.
      - Convert geometry to a single scalar rho = 1 - effective_rank(UᵀU).
      - Allocate each role's budget: diagonals + rho structured + random coverage.

    Returns:
      pairs_by_role: {"action":[(i,j),...], "out":[...]} each of length J_role
      U:             The sketch matrix (D,k) used for diagnostics/visualizations
      diag:          HeadRKDiagnostics with geometry + graph stats
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device

    D = int(d_model or getattr(model, "d_model", 0))
    if D <= 0:
        raise ValueError("Provide d_model explicitly or expose model.d_model > 0.")
    if min(budgets.J_out, budgets.J_action, budgets.n_self) < 0:
        raise ValueError("Budgets must be non-negative.")
    if budgets.n_self > min(budgets.J_out, budgets.J_action):
        raise ValueError("n_self must be <= each role budget under random-pairing.")

    k = choose_sketch_dim(D=D, J=max(1, budgets.J_out, budgets.J_action))

    # Determine warmup_batches from target observations if not specified.
    # We estimate observations per batch once we see the first trace shape.
    sketch, n_obs, rho_windows = TraceSketch.from_warmup(
        D=D,
        k=k,
        seed=warmup.seed,
        model=model,
        dataloader=dataloader,
        trace_fn=trace_fn,
        warmup_batches=warmup.warmup_batches,
        target_observations=warmup.target_observations,
        max_warmup_batches=warmup.max_warmup_batches,
        sampling=warmup.sampling,
        device=device,
        batch_to_x=batch_to_x,
        warmup_context=warmup_context,
        preserve_rng=warmup.preserve_global_rng_state,
    )

    rho, eff_rank = KernelGeometry(sketch.U).rho_and_effective_rank()

    g = torch.Generator(device=device)
    g.manual_seed(int(warmup.seed) + 13_371)

    shared = PairSet(D=D) if disjoint_roles else None

    out_pairs, out_stats = PairInitializer(D=D, rho=rho, gen=g).for_role(
        J=budgets.J_out, n_self=budgets.n_self, U=sketch.U, shared_seen=shared
    )
    act_pairs, act_stats = PairInitializer(D=D, rho=rho, gen=g).for_role(
        J=budgets.J_action, n_self=budgets.n_self, U=sketch.U, shared_seen=shared
    )

    overlap_pairs, overlap_nodes = _pairset_overlap_stats(out_pairs, act_pairs)

    diag = HeadRKDiagnostics(
        D=D,
        k=k,
        rho=float(rho),
        eff_rank=float(eff_rank),
        n_obs=int(n_obs),
        rho_window_mean=float(rho_windows.mean().item()) if rho_windows.numel() else float("nan"),
        rho_window_std=float(rho_windows.std(unbiased=False).item()) if rho_windows.numel() else float("nan"),
        out=out_stats,
        action=act_stats,
        overlap_pairs=float(overlap_pairs),
        overlap_nodes=float(overlap_nodes),
    )

    return {"out": out_pairs, "action": act_pairs}, sketch.U, diag


# =============================================================================
# Warmup trace sketch  U = ZR
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TraceSketch:
    """
    Streaming JL sketch of the trace matrix Z.

    We conceptually stack per-neuron trace vectors Z_i over warmup observations (ticks & samples):
        Z ∈ R^{D×N},   S = ZZᵀ is the true Gram matrix of neuron trace synchrony.

    We avoid O(D²) by sketching:
        U = ZR,   R ∈ R^{N×k} has Rademacher(±1)/sqrt(k) entries.
    Then ⟨Z_i, Z_j⟩ ≈ ⟨U_i, U_j⟩ and we can reason about geometry through U.

    Implementation detail:
      We never form Z. For a batch of observations z ∈ R^{n×D} we sample r ∈ R^{n×k}
      and accumulate:
          U += zᵀ r
    """

    D: int
    k: int
    U: Tensor

    @staticmethod
    @torch.no_grad()
    def from_warmup(
        *,
        D: int,
        k: int,
        seed: int,
        model: nn.Module,
        dataloader: Iterable,
        trace_fn: Callable[[nn.Module, Tensor], Tensor],
        warmup_batches: Optional[int],
        target_observations: int,
        max_warmup_batches: int,
        sampling: str,
        device: torch.device,
        batch_to_x: Callable,
        warmup_context: contextlib.AbstractContextManager,
        preserve_rng: bool,
    ) -> Tuple["TraceSketch", int, Tensor]:
        U = torch.zeros((D, k), device=device, dtype=torch.float32)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        devices = [device.index] if device.type == "cuda" and device.index is not None else None
        fork = torch.random.fork_rng(devices=devices, enabled=bool(preserve_rng))

        it: Iterator = iter(dataloader)

        # We compute rho on windows to assess warmup stability.
        rho_windows: List[float] = []
        n_obs_total = 0

        def take_next_batch() -> object:
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(dataloader)
                return next(it)

        def skip_batches(n: int) -> None:
            for _ in range(max(0, int(n))):
                _ = take_next_batch()

        sampling = str(sampling).lower()
        if sampling not in {"contiguous", "spaced"}:
            raise ValueError("warmup.sampling must be 'contiguous' or 'spaced'.")

        # If warmup_batches is None, we choose it after observing the first trace shape.
        chosen_batches = int(warmup_batches) if warmup_batches is not None else None

        # A simple spaced scheme: spread N batches over ~4N batches by skipping ~3 each time.
        # This is intentionally conservative and data-only (no training).
        def warmup_stride(n_batches: int) -> int:
            if sampling == "contiguous":
                return 0
            return max(1, 3)  # fixed stride keeps mental model simple

        stride = 0

        with fork, warmup_context, torch.inference_mode():
            # One probe batch to infer observations-per-batch if needed.
            probe = take_next_batch()
            x = batch_to_x(probe).to(device, non_blocking=True)
            z = flatten_ticks(trace_fn(model, x)).to(device=device, dtype=torch.float32, non_blocking=True)
            if z.shape[-1] != D:
                raise ValueError(f"trace_fn produced D={z.shape[-1]} but expected D={D}")
            obs_per_batch = int(z.shape[0])
            if obs_per_batch <= 0:
                raise ValueError("trace_fn produced an empty trace batch.")

            if chosen_batches is None:
                chosen_batches = int(math.ceil(target_observations / max(1, obs_per_batch)))
                chosen_batches = min(max(1, chosen_batches), int(max_warmup_batches))

            stride = warmup_stride(chosen_batches)

            # Now process the probe as batch #1.
            for batch_index in range(chosen_batches):
                if batch_index > 0:
                    if stride:
                        skip_batches(stride)
                    probe = take_next_batch()
                    x = batch_to_x(probe).to(device, non_blocking=True)
                    z = flatten_ticks(trace_fn(model, x)).to(device=device, dtype=torch.float32, non_blocking=True)

                # Per-observation RMS normalization makes the sketch reflect correlation geometry
                # rather than raw scale drift. NaN/Inf are neutralized to keep initialization safe.
                z = z / (reduce(z * z, "n d -> n 1", "mean").add(1e-8).sqrt())
                z = torch.nan_to_num(z)

                r = rademacher((z.shape[0], k), device=device, gen=gen) / math.sqrt(k)
                U.add_(z.T @ r)

                n_obs_total += int(z.shape[0])

                # Windowed rho diagnostics every ~4 batches (cheap, stable).
                if (batch_index + 1) % 4 == 0:
                    rho_w, _ = KernelGeometry(U).rho_and_effective_rank()
                    rho_windows.append(float(rho_w))

        return TraceSketch(D=D, k=k, U=U), n_obs_total, torch.tensor(rho_windows, device="cpu", dtype=torch.float32)


def flatten_ticks(z: Tensor) -> Tensor:
    """
    Normalize trace_fn outputs to a 2D observation matrix.

      (B, D)     -> (B, D)
      (B, T, D)  -> (B*T, D)

    Here "ticks are observations": each internal step adds a row to Z.
    """
    if z.ndim == 2:
        return z
    if z.ndim == 3:
        return rearrange(z, "b t d -> (b t) d")
    raise ValueError(f"trace_fn must return (B,D) or (B,T,D); got {tuple(z.shape)}")


def rademacher(shape: Tuple[int, ...], *, device: torch.device, gen: torch.Generator) -> Tensor:
    """
    Rademacher(±1) matrix.

    Rademacher projections are a standard fast alternative to Gaussian JL matrices:
      - entries are mean 0, variance 1
      - cheap to sample (integers -> +/-1)
      - inner products concentrate similarly to Gaussian in practice
    """
    x = torch.randint(0, 2, shape, device=device, generator=gen, dtype=torch.int8)
    return (x * 2 - 1).to(torch.float32)


# =============================================================================
# Geometry scalar rho via effective rank of UᵀU
# =============================================================================

@dataclasses.dataclass(frozen=True)
class KernelGeometry:
    U: Tensor

    def rho_and_effective_rank(self) -> Tuple[float, float]:
        """
        We summarize geometry through the (k×k) Gram matrix:
          G = UᵀU

        Let {λ} be eigenvalues of G (nonnegative). Participation-ratio effective rank:
          r_eff = (∑λ)^2 / (∑λ^2)

        Normalize to [0,1] by dividing by k:
          eff_rank = r_eff / k = (∑λ)^2 / (k ∑λ^2)

        Then:
          rho = 1 - eff_rank

        Interpretation:
          - eff_rank ~ 1  => spectrum spread => diffuse geometry => random-like
          - eff_rank small => low-rank      => hub-like / structured
        """
        G = self.U.T @ self.U
        eigs = torch.linalg.eigvalsh(G).clamp_min(0)

        s1 = eigs.sum()
        s2 = (eigs * eigs).sum()
        k = eigs.numel()

        if float(s1) <= 0.0 or float(s2) <= 0.0:
            eff = 0.0
        else:
            eff = float((s1 * s1 / (k * s2)).clamp(0, 1).item())

        return float(1.0 - eff), float(eff)


def choose_sketch_dim(*, D: int, J: int) -> int:
    """
    Deterministic sketch resolution with no arbitrary caps.

    k must be large enough to:
      - estimate rho stably
      - rank candidate edges by |<U_i, U_j>| robustly

    A simple, monotone rule that grows with budget J and width D:
      k = min(D, ceil( sqrt(J) + log2(D+1)^2 ))
    """
    return min(int(D), max(1, int(math.ceil(math.sqrt(float(J)) + math.log2(float(D) + 1.0) ** 2))))


# =============================================================================
# Pair set & sampling
# =============================================================================

class PairSet:
    """
    Ordered, de-duplicating set of unordered pairs (i<=j).

    Correctness constraints:
      - Self pairs must appear first (prefix contract).
      - Structured edges should follow.
      - Random tail fills remaining, preserving append order.

    torch.unique() sorts and breaks this contract, so we use a Python set for membership
    and a list for stable order. Budgets are small enough for this to be clean and safe.
    """
    def __init__(self, *, D: int):
        self.D = int(D)
        self._seen: set[int] = set()
        self._codes: List[int] = []

    def __len__(self) -> int:
        return len(self._codes)

    @property
    def seen_codes(self) -> set[int]:
        return self._seen

    def add_pair(self, i: int, j: int) -> bool:
        a, b = (i, j) if i <= j else (j, i)
        code = a * self.D + b
        if code in self._seen:
            return False
        self._seen.add(code)
        self._codes.append(code)
        return True

    def add_diag(self, nodes: Tensor) -> int:
        added = 0
        for i in nodes.tolist():
            added += int(self.add_pair(int(i), int(i)))
        return added

    def add_scored_edges(self, edges: Tensor, scores: Tensor, limit: int) -> int:
        if limit <= 0 or edges.numel() == 0:
            return 0
        added = 0
        for idx in torch.argsort(scores, descending=True).tolist():
            if added >= limit:
                break
            i, j = map(int, edges[idx].tolist())
            added += int(self.add_pair(i, j))
        return added

    def fill_uniform_offdiag(self, n: int, *, gen: torch.Generator, device: torch.device) -> int:
        """
        Fill with *uniform unordered off-diagonal* pairs (i<j), stably.

        We sample uniformly from the upper triangle without diagonal:
          total = D(D-1)/2
          idx ~ Uniform{0,...,total-1}
        Then map idx -> (i,j) using an inverse triangular-number transform.

        This vectorizes the expensive part (sampling/mapping) while keeping stable insertion
        (append order + de-dup) in a small Python loop.
        """
        if n <= 0 or self.D < 2:
            return 0

        total = self.D * (self.D - 1) // 2
        need, added = int(n), 0

        def idx_to_pair(idx: Tensor) -> Tuple[Tensor, Tensor]:
            # Enumerate pairs by rows:
            # row i contains (i, i+1..D-1), count = D-1-i
            # prefix T(i) = i*(2D-i-1)/2
            idx_f = idx.to(torch.float64)
            twoD_1 = float(2 * self.D - 1)
            disc = (twoD_1 * twoD_1 - 8.0 * idx_f).clamp_min(0.0)
            i = torch.floor((twoD_1 - torch.sqrt(disc)) / 2.0).to(torch.int64).clamp(0, self.D - 2)
            Ti = (i * (2 * self.D - i - 1)) // 2
            off = (idx - Ti).to(torch.int64)
            j = i + 1 + off
            # rare numeric correction
            bad = j >= self.D
            if bad.any():
                i2 = (i[bad] - 1).clamp(0, self.D - 2)
                Ti2 = (i2 * (2 * self.D - i2 - 1)) // 2
                off2 = (idx[bad] - Ti2).to(torch.int64)
                j2 = i2 + 1 + off2
                i = i.clone()
                j = j.clone()
                i[bad], j[bad] = i2, j2
            return i, j

        for _ in range(8):  # bounded refill loop
            if need <= 0:
                break
            m = max(1024, 8 * need)
            idx = torch.randint(0, total, (m,), device=device, generator=gen)
            i, j = idx_to_pair(idx)
            codes = (i * self.D + j).to(torch.int64)  # i<j

            for c in codes.tolist():
                if need <= 0:
                    break
                if c in self._seen:
                    continue
                self._seen.add(int(c))
                self._codes.append(int(c))
                need -= 1
                added += 1

        # deterministic completion for tiny D
        if need > 0 and self.D <= 4:
            for i in range(self.D - 1):
                for j in range(i + 1, self.D):
                    if need <= 0:
                        break
                    if self.add_pair(i, j):
                        need -= 1
                        added += 1

        return added

    def pairs(self) -> List[Pair]:
        D = self.D
        return [(c // D, c % D) for c in self._codes]


@dataclasses.dataclass(frozen=True)
class PairInitializer:
    """
    Construct role pairs: diag + structured + random.

    rho converts sketch geometry into a single "how structured is it?" scalar.
    We allocate structured_budget = round(rho * remaining) and fill the rest uniformly.

    Structured edges are built as a sparse multi-hub graph rather than a clique:
      - candidate nodes by energy ||U_i||²
      - hubs by row-mass in |Uc Ucᵀ|
      - neighbors by top weights per hub
      - edges scored by |<U_i,U_j>|; keep the top E
    """
    D: int
    rho: float
    gen: torch.Generator

    def for_role(
        self,
        *,
        J: int,
        n_self: int,
        U: Tensor,
        shared_seen: Optional[PairSet],
    ) -> Tuple[List[Pair], Dict[str, float]]:
        if J <= 0:
            return [], {"J": 0, "self": 0, "struct": 0, "rand": 0, "nodes_frac": 0.0, "deg_gini": 0.0, "deg_max_over_mean": 0.0, "top10_edge_share": 0.0}

        role = PairSet(D=self.D)
        if shared_seen is not None:
            role.seen_codes.update(shared_seen.seen_codes)

        node_energy = reduce(U * U, "d k -> d", "sum")

        # 1) diagonals (prefix)
        n_diag = min(int(n_self), int(J))
        diag_nodes = torch.topk(node_energy, k=n_diag).indices if n_diag else torch.empty((0,), device=U.device, dtype=torch.long)
        self_added = role.add_diag(diag_nodes)

        # 2) structured budget
        remaining = J - len(role)
        struct_budget = int(round(self.rho * remaining)) if remaining > 0 else 0
        struct_budget = max(0, min(remaining, struct_budget))
        struct_added = self._add_structured(role=role, U=U, node_energy=node_energy, E=struct_budget)

        # 3) random coverage
        rand_added = role.fill_uniform_offdiag(J - len(role), gen=self.gen, device=U.device)

        if shared_seen is not None:
            shared_seen.seen_codes.update(role.seen_codes)

        pairs = role.pairs()[:J]
        stats = role_graph_stats(D=self.D, pairs=pairs, n_self=n_diag)
        return pairs, {
            "J": float(J),
            "self": float(self_added),
            "struct": float(struct_added),
            "rand": float(rand_added),
            **stats,
        }

    def _add_structured(self, *, role: PairSet, U: Tensor, node_energy: Tensor, E: int) -> int:
        if E <= 0 or self.D < 2:
            return 0

        M = min(self.D, max(2, 2 * int(math.ceil(math.sqrt(2.0 * E)))))
        candidates = torch.topk(node_energy, k=M).indices
        Uc = U[candidates]  # (M,k)

        W = (Uc @ Uc.T).abs()
        W = W.masked_fill(torch.eye(M, device=W.device, dtype=torch.bool), float("-inf"))

        hub_strength = reduce(W, "m n -> m", "sum")
        H = min(M, max(1, int(math.ceil(math.sqrt(float(E))))))
        hubs = torch.topk(hub_strength, k=H).indices  # (H,)

        L = min(M - 1, max(1, int(math.ceil(float(E) / float(H)))))
        neighbors = torch.topk(W[hubs], k=L, dim=1).indices  # (H,L)

        hub_nodes = candidates[hubs]
        hub_nodes = repeat(hub_nodes, "h -> h l", l=L)
        neighbor_nodes = candidates[neighbors]

        edges = rearrange(torch.stack([hub_nodes, neighbor_nodes], dim=-1), "h l two -> (h l) two")
        scores = rearrange(W[hubs].gather(1, neighbors), "h l -> (h l)")

        return role.add_scored_edges(edges=edges, scores=scores, limit=E)


# =============================================================================
# Diagnostics & visualizations
# =============================================================================

def log_headrk_initialization(
    *,
    log_dir: str,
    U: Tensor,
    pairs_by_role: Dict[str, List[Pair]],
    diag: HeadRKDiagnostics,
    max_nodes_heatmap: int = 512,
    kernel_heatmap_nodes: int = 256,
) -> None:
    """
    Write a compact but high-signal diagnostic report and plots.

    Plots are intentionally simple and robust:
      - adjacency heatmap on top nodes by degree (sorted), per role
      - degree curve per role
      - optional kernel proxy heatmap |Uc Ucᵀ| on top-energy nodes

    This function does not require seaborn; it uses matplotlib only.
    """
    os.makedirs(log_dir, exist_ok=True)

    # 1) text report
    report_path = os.path.join(log_dir, "headrk_init.txt")
    with open(report_path, "w") as f:
        print("HEADR-K initialization diagnostics", file=f)
        print(f"D={diag.D}  k={diag.k}", file=f)
        print(f"rho={diag.rho:.4f}  eff_rank={diag.eff_rank:.4f}  n_obs={diag.n_obs}", file=f)
        print(f"rho_windows mean±std = {diag.rho_window_mean:.4f} ± {diag.rho_window_std:.4f}", file=f)
        print(f"pair overlap: pairs={diag.overlap_pairs:.4f}  nodes={diag.overlap_nodes:.4f}", file=f)
        print("", file=f)
        for role in ("action", "out"):
            st = diag.action if role == "action" else diag.out
            print(f"[{role}]", file=f)
            for k, v in st.items():
                print(f"  {k}: {v}", file=f)
            print("", file=f)

    # 2) plots
    import matplotlib.pyplot as plt

    for role, pairs in pairs_by_role.items():
        stats = role_graph_stats(D=diag.D, pairs=pairs, n_self=int(min(len(pairs), int(diag.action.get("self", 0) if role == "action" else diag.out.get("self", 0)))))
        deg = stats["_deg"].cpu()
        order = torch.argsort(deg, descending=True)

        # Degree curve
        plt.figure()
        plt.plot(deg[order].numpy())
        plt.title(f"HEADR-K degree curve ({role})")
        plt.xlabel("node rank (sorted by degree)")
        plt.ylabel("degree")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"headrk_degree_{role}.png"), dpi=150)
        plt.close()

        # Adjacency heatmap for top nodes by degree
        top = order[: min(max_nodes_heatmap, diag.D)]
        idx = {int(n): i for i, n in enumerate(top.tolist())}

        A = torch.zeros((len(top), len(top)), dtype=torch.float32)
        for i, j in pairs:
            if i == j:
                continue
            if int(i) in idx and int(j) in idx:
                a, b = idx[int(i)], idx[int(j)]
                A[a, b] = 1.0
                A[b, a] = 1.0

        plt.figure(figsize=(6, 6))
        plt.imshow(A.numpy(), interpolation="nearest", aspect="auto")
        plt.title(f"HEADR-K adjacency (top-{len(top)} by degree) ({role})")
        plt.xlabel("nodes (sorted by degree)")
        plt.ylabel("nodes (sorted by degree)")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"headrk_adj_{role}.png"), dpi=150)
        plt.close()

    # Kernel proxy heatmap on top-energy nodes (role-agnostic, explains geometry)
    if U is not None and U.numel():
        node_energy = reduce(U * U, "d k -> d", "sum").detach().cpu()
        top = torch.topk(node_energy, k=min(kernel_heatmap_nodes, node_energy.numel())).indices
        Uc = U.detach().cpu()[top]
        K = (Uc @ Uc.T).abs()
        plt.figure(figsize=(6, 6))
        plt.imshow(K.numpy(), interpolation="nearest", aspect="auto")
        plt.title(f"HEADR-K kernel proxy |Uc Ucᵀ| (top-{len(top)} energy)")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "headrk_kernel_proxy.png"), dpi=150)
        plt.close()


def role_graph_stats(*, D: int, pairs: Sequence[Pair], n_self: int) -> Dict[str, float]:
    """
    Role graph stats for "what prior did we impose?"

    We treat the pair set as an undirected graph on nodes 0..D-1.
    Self pairs are ignored for degree stats; they are a separate "snapshot" channel.

    Returns human-facing scalars plus a private '_deg' tensor used by plotting.
    """
    deg = torch.zeros((D,), dtype=torch.float32)
    used = torch.zeros((D,), dtype=torch.bool)

    # Track whether prefix contract holds (first n_self diagonals).
    prefix_ok = True
    for t, (i, j) in enumerate(pairs[:n_self]):
        prefix_ok = prefix_ok and (i == j)

    off_edges = 0
    for (i, j) in pairs:
        used[int(i)] = True
        used[int(j)] = True
        if i == j:
            continue
        deg[int(i)] += 1
        deg[int(j)] += 1
        off_edges += 1

    nodes_frac = float(used.float().mean().item())
    deg_mean = float(deg.mean().item())
    deg_max = float(deg.max().item())
    deg_max_over_mean = float(deg_max / (deg_mean + 1e-12))

    gini = float(gini_coefficient(deg))

    # Share of edges incident to top-10 nodes (hub mass proxy)
    top10 = torch.topk(deg, k=min(10, D)).values.sum().item()
    top10_edge_share = float(top10 / (2.0 * max(1, off_edges)))  # each edge contributes 2 to degree sum

    return {
        "nodes_frac": nodes_frac,
        "deg_gini": gini,
        "deg_max_over_mean": deg_max_over_mean,
        "top10_edge_share": top10_edge_share,
        "prefix_self_ok": float(prefix_ok),
        "_deg": deg,
    }


def gini_coefficient(x: Tensor) -> float:
    """
    Gini coefficient for nonnegative values.

    Interpretation:
      0   => perfectly uniform degrees (random-like)
      1   => all mass on one node (pure hub)
    """
    x = x.detach().float().clamp_min(0).cpu()
    if x.numel() == 0:
        return 0.0
    if float(x.sum()) <= 0.0:
        return 0.0
    xs = torch.sort(x)[0]
    n = xs.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    g = (2.0 * (idx * xs).sum() / (n * xs.sum()) - (n + 1.0) / n).item()
    return float(max(0.0, min(1.0, g)))


def _pairset_overlap_stats(a: Sequence[Pair], b: Sequence[Pair]) -> Tuple[float, float]:
    def canon(p: Pair) -> Pair:
        return (p[0], p[1]) if p[0] <= p[1] else (p[1], p[0])

    A = set(map(canon, a))
    B = set(map(canon, b))
    pair_overlap = len(A & B) / max(1, min(len(A), len(B)))

    nodesA = set([i for p in A for i in p])
    nodesB = set([i for p in B for i in p])
    node_overlap = len(nodesA & nodesB) / max(1, min(len(nodesA), len(nodesB)))

    return float(pair_overlap), float(node_overlap)
