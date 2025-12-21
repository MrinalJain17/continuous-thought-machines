from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch


Tensor = torch.Tensor


def _stable_hash_pairs(left: Tensor, right: Tensor) -> str:
    # Stable fingerprint across runs (device-agnostic).
    a = left.detach().to("cpu", torch.int64).numpy().tobytes()
    b = right.detach().to("cpu", torch.int64).numpy().tobytes()
    return hashlib.sha256(a + b).hexdigest()[:12]


def _degree_stats(left: np.ndarray, right: np.ndarray) -> Tuple[int, float, int, int]:
    # undirected degrees on the induced endpoint set
    deg = {}
    for u, v in zip(left.tolist(), right.tolist()):
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    vals = np.array(list(deg.values()), dtype=np.int32)
    if vals.size == 0:
        return 0, 0.0, 0, 0
    return int(vals.min()), float(np.median(vals)), int(vals.max()), int(vals.size)


def _top_frac_incident(left: np.ndarray, right: np.ndarray, top_frac: float = 0.01) -> float:
    # fraction of edges incident to top-degree nodes (hub concentration)
    deg = {}
    for u, v in zip(left.tolist(), right.tolist()):
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if not deg:
        return 0.0
    nodes = np.array(list(deg.keys()), dtype=np.int64)
    dvals = np.array([deg[n] for n in nodes], dtype=np.int64)
    k = max(1, int(np.ceil(top_frac * len(nodes))))
    top_nodes = set(nodes[np.argsort(dvals)[::-1][:k]].tolist())
    incident = 0
    for u, v in zip(left.tolist(), right.tolist()):
        if u in top_nodes or v in top_nodes:
            incident += 1
    return float(incident / max(1, left.size))


def _scale_free_near0(x: Tensor, frac: float = 1e-2, eps: float = 1e-12) -> float:
    # near0 = P(|x| < frac * RMS(x)), dimensionless
    xrms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    thr = frac * xrms
    return float((x.abs() < thr).float().mean().item())


def _scale_free_sat(x: Tensor, mult: float = 3.0, eps: float = 1e-12) -> float:
    # saturation proxy = P(|x| > mult * RMS(x)), dimensionless
    xrms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    thr = mult * xrms
    return float((x.abs() > thr).float().mean().item())


@dataclass(frozen=True)
class PairGraphStats:
    J: int
    n_self: int
    self_prefix_frac: float
    self_outside_prefix_frac: float
    self_frac: float
    unique_pairs_frac: float
    unique_neurons: int
    deg_min: int
    deg_med: float
    deg_max: int
    top1pct_incident: float
    fingerprint: str


@dataclass(frozen=True)
class HealthStats:
    z_rms: float
    action_r: float
    out_r: float
    action_near0: float
    out_near0: float
    action_sat: float
    out_sat: float
    cos_action_out: float
    t_star: Optional[int]


class CTMStatsCollector:
    """Logs (1) init pairing diagnostics and (2) periodic sync health stats.

    This is intentionally task-agnostic: it only needs tensors already produced by CTM.
    """

    def __init__(self, log_fn=print):
        self.log_fn = log_fn

    # -------------------- Init / graph diagnostics --------------------

    def pairing_graph_stats(self, role: str, left: Tensor, right: Tensor, n_self: int) -> PairGraphStats:
        left_cpu = left.detach().to("cpu", torch.int64).numpy()
        right_cpu = right.detach().to("cpu", torch.int64).numpy()
        J = int(left_cpu.shape[0])
        n_self = int(min(n_self, J))

        pairs = list(zip(left_cpu.tolist(), right_cpu.tolist()))
        unique_pairs_frac = len(set(pairs)) / max(1, J)

        # Self fractions (make the contract visible)
        if n_self > 0:
            self_prefix_frac = float(np.mean(left_cpu[:n_self] == right_cpu[:n_self]))
        else:
            self_prefix_frac = 0.0

        if J > n_self:
            self_outside_prefix_frac = float(np.mean(left_cpu[n_self:] == right_cpu[n_self:]))
        else:
            self_outside_prefix_frac = 0.0

        self_frac = float(np.mean(left_cpu == right_cpu))

        deg_min, deg_med, deg_max, unique_neurons = _degree_stats(left_cpu, right_cpu)
        top1 = _top_frac_incident(left_cpu, right_cpu, top_frac=0.01)
        fp = _stable_hash_pairs(left, right)

        return PairGraphStats(
            J=J,
            n_self=n_self,
            self_prefix_frac=self_prefix_frac,
            self_outside_prefix_frac=self_outside_prefix_frac,
            self_frac=self_frac,
            unique_pairs_frac=float(unique_pairs_frac),
            unique_neurons=int(unique_neurons),
            deg_min=int(deg_min),
            deg_med=float(deg_med),
            deg_max=int(deg_max),
            top1pct_incident=float(top1),
            fingerprint=fp,
        )

    def log_init(
        self,
        warmup_diag: Dict[str, Any],
        role_graph: Dict[str, PairGraphStats],
        role_policy: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        warmup_diag: output of initialize_ctm_pairs() top-level fields
        role_graph: {"action": PairGraphStats, "out": PairGraphStats}
        role_policy: {"action": diag["action"], "out": diag["out"]} (the rich init diag dicts)
        """
        self.log_fn("=== CTM Pair Init Summary ===")
        self.log_fn(
            f"Warmup: batches={warmup_diag.get('warmup_batches')} "
            f"res_cap={warmup_diag.get('reservoir_capacity')} "
            f"res_action={warmup_diag.get('reservoir_size_action')}/{warmup_diag.get('reservoir_seen_action')} "
            f"res_out={warmup_diag.get('reservoir_size_out')}/{warmup_diag.get('reservoir_seen_out')} "
            f"t_star_mean={warmup_diag.get('t_star_mean'):.2f} "
            f"t_star_std={warmup_diag.get('t_star_std'):.2f}"
        )

        for role in ["action", "out"]:
            st = role_graph[role]
            pol = role_policy[role]

            # Low-rank + policy diagnostics (the ones you said were “missing”)
            k_eff = pol.get("k_eff", warmup_diag.get(role, {}).get("k_eff", -1))
            eta = pol.get("eta", warmup_diag.get(role, {}).get("eta", float("nan")))

            self.log_fn(
                f"[{role}] "
                f"J={st.J} n_self={st.n_self} "
                f"self_prefix={st.self_prefix_frac:.3f} self_outside={st.self_outside_prefix_frac:.3f} self_total={st.self_frac:.3f} "
                f"unique_pairs={st.unique_pairs_frac:.3f} "
                f"unique_neurons={st.unique_neurons} "
                f"deg(min/med/max)={st.deg_min}/{st.deg_med:.1f}/{st.deg_max} "
                f"top1%_incident={st.top1pct_incident:.3f} "
                f"k_eff={int(k_eff)} eta={float(eta):.3f} "
                f"M={int(pol.get('M', -1))} n_eff={float(pol.get('n_eff', float('nan'))):.1f} "
                f"f={float(pol.get('f_bottleneck', float('nan'))):.3f} "
                f"J_core={int(pol.get('J_core', -1))} J_wide={int(pol.get('J_wide', -1))} "
                f"Bn={int(pol.get('Bn', -1))} dmax_core={int(pol.get('dmax_core', -1))} dmax_wide={int(pol.get('dmax_wide', -1))} "
                f"fp={st.fingerprint}"
            )

    def export_gephi_csv(self, path: str, role: str, left: Tensor, right: Tensor) -> None:
        # Minimal Gephi edge list.
        l = left.detach().to("cpu", torch.int64).numpy()
        r = right.detach().to("cpu", torch.int64).numpy()
        with open(path, "w", encoding="utf-8") as f:
            f.write("source,target,role,weight\n")
            for u, v in zip(l.tolist(), r.tolist()):
                f.write(f"{u},{v},{role},1\n")

    # -------------------- Training health diagnostics --------------------

    def compute_health(
        self,
        z: Tensor,
        s_action: Tensor,
        s_out: Tensor,
        t_star: Optional[int] = None,
        eps: float = 1e-12,
    ) -> HealthStats:
        # z: (B,D), s_action/s_out: (B,J_role)
        z_rms = float(torch.sqrt((z * z).mean() + eps).item())

        def ratio(sync: Tensor) -> float:
            num = torch.sqrt((sync * sync).sum(dim=-1) + eps)
            den = torch.sqrt((z * z).sum(dim=-1) + eps)
            return float((num / den).mean().item())

        # cosine between the two sync spaces (aggregate per batch)
        # project to a scalar via mean pairwise cosine along batch:
        a = s_action
        o = s_out
        # Align dims by reducing to per-sample norms/cos with dot after padding to min dim
        m = min(a.shape[-1], o.shape[-1])
        adot = (a[..., :m] * o[..., :m]).sum(dim=-1)
        an = torch.sqrt((a[..., :m] * a[..., :m]).sum(dim=-1) + eps)
        on = torch.sqrt((o[..., :m] * o[..., :m]).sum(dim=-1) + eps)
        cos = float((adot / (an * on)).mean().item())

        return HealthStats(
            z_rms=z_rms,
            action_r=ratio(s_action),
            out_r=ratio(s_out),
            action_near0=_scale_free_near0(s_action),
            out_near0=_scale_free_near0(s_out),
            action_sat=_scale_free_sat(s_action),
            out_sat=_scale_free_sat(s_out),
            cos_action_out=cos,
            t_star=t_star,
        )

    def log_health(self, step: int, hs: HealthStats) -> None:
        self.log_fn(
            f"[step {step}] z_rms={hs.z_rms:.4f} "
            f"r_action={hs.action_r:.4f} r_out={hs.out_r:.4f} "
            f"near0_action={hs.action_near0:.3f} near0_out={hs.out_near0:.3f} "
            f"sat_action={hs.action_sat:.3f} sat_out={hs.out_sat:.3f} "
            f"cos(a,o)={hs.cos_action_out:.3f} "
            f"t_star={hs.t_star}"
        )


class SyncTickCollector:
    def __init__(self):
        self.action = {}
        self.out = {}

    def reset(self):
        self.action.clear()
        self.out.clear()

    def __call__(self, tick_idx: int, role: str, sync_vec: torch.Tensor):
        if role == "action":
            self.action[tick_idx] = sync_vec.detach()
        else:
            self.out[tick_idx] = sync_vec.detach()

    def get(self, role: str, tick_idx: int):
        d = self.action if role == "action" else self.out
        if tick_idx in d:
            return d[tick_idx]
        # fallback to last available
        if len(d) == 0:
            return None
        return d[max(d.keys())]
