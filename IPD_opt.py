import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sqlalchemy.orm.persistence import post_update


def _safe_mean(x_sum, x_cnt):
    if x_cnt == 0:
        return None
    return x_sum / x_cnt


def _compute_full_grad_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total_sq += float((g * g).sum().item())
    return math.sqrt(total_sq)


def _compute_param_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            w = p.detach()
            total_sq += float((w * w).sum().item())
    return math.sqrt(total_sq)


def _snapshot_init_params_cpu(model):
    return {
        id(p): p.detach().cpu().clone()
        for p in model.parameters()
        if p.requires_grad
    }


def _compute_delta_from_init_cpu(model, init_params_cpu):
    total_sq = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p0 = init_params_cpu.get(id(p), None)
        if p0 is None:
            continue
        cur = p.detach().cpu()
        diff = cur - p0
        total_sq += float((diff * diff).sum().item())
    return math.sqrt(total_sq)


def _compute_E_diff_proxy(model, optimizer):
    """
    只在 enable_diff=True 时计算一个与 diff 构造一致的结构能量 proxy:
        E_diff = mean( Laplacian(p)^2 )
    """
    if not hasattr(optimizer, "ipd_config"):
        return None
    if not optimizer.ipd_config.get("enable_diff", False):
        return None
    if not hasattr(optimizer, "map"):
        return None
    if not hasattr(optimizer, "_lap2d_tensor"):
        return None
    if not hasattr(optimizer, "_lap1d_row"):
        return None

    cfg = optimizer.ipd_config
    lap_r = cfg.get("lap_r", 1)
    lap_neigh = cfg.get("lap_neigh", "l1")
    lap_padding = cfg.get("lap_padding", "reflect")

    total_sq = 0.0
    total_n = 0

    for p in model.parameters():
        if not p.requires_grad:
            continue

        info = optimizer.map.get(id(p), {"type": "none"})
        layer_type = info.get("type", "none")

        raw = None
        if layer_type == "intra_kernel_conv" and p.dim() == 4:
            raw = optimizer._lap2d_tensor(
                p.detach(), lap_r, lap_neigh, lap_padding
            )
        elif layer_type == "intra_vector_fc" and p.dim() == 2:
            raw = optimizer._lap1d_row(
                p.detach(), lap_r, lap_padding
            )

        if raw is None:
            continue

        total_sq += float((raw * raw).sum().item())
        total_n += raw.numel()

    if total_n == 0:
        return None
    return total_sq / total_n


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    else:
        return obj
class IPD(torch.optim.Optimizer):
    def __init__(
        self,
        base_optimizer,
        total_steps=10000,
        diff_decay_ratio=0.2,
        coul_decay_ratio=0.3,

        map=None,
        enable_diff=False, r_diff=0.0,
        enable_coul=False, r_coul=0.0,

        cap_mult=1.0,

        lap_r=1,
        lap_neigh='l1',
        lap_padding='reflect',
        diff_mix_perp=1.0,

        debug_first_n_steps=0,
    ):
        """
        IPD Meta-Optimizer

        当前版本统一采用 coupled / pre 注入：
            p.grad <- p.grad - F_total
            然后交给 base_optimizer.step()
        """
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults

        # 不污染 base optimizer 的 state
        self.state = {}
        self.ipd_state = {}

        self.global_step = 0
        self.debug_first_n_steps = debug_first_n_steps

        self.map = map if map is not None else {}
        self.ipd_config = dict(
            total_steps=total_steps,
            diff_decay_ratio=diff_decay_ratio,
            coul_decay_ratio=coul_decay_ratio,
            enable_diff=enable_diff,
            r_diff=r_diff,
            enable_coul=enable_coul,
            r_coul=r_coul,
            cap_mult=cap_mult,
            lap_r=lap_r,
            lap_neigh=lap_neigh,
            lap_padding=lap_padding,
            diff_mix_perp=diff_mix_perp,
        )

        self._kern_cache = {}
        self.reset_ipd_stats()

    # -----------------------------
    # private state
    # -----------------------------
    def _get_ipd_state(self, p):
        pid = id(p)
        if pid not in self.ipd_state:
            self.ipd_state[pid] = {}
        return self.ipd_state[pid]

    # -----------------------------
    # kernel builders
    # -----------------------------
    def _get_lap1d_kernel(self, r, device, dtype):
        key = ('1d', r, device, dtype)
        if key not in self._kern_cache:
            k = torch.zeros(2 * r + 1, device=device, dtype=dtype)
            k[r] = -2 * r
            k[:r] = 1.0
            k[r + 1:] = 1.0
            self._kern_cache[key] = k.view(1, 1, -1)
        return self._kern_cache[key]

    def _get_lap2d_kernel(self, r, neighborhood, device, dtype):
        key = ('2d', r, neighborhood, device, dtype)
        if key not in self._kern_cache:
            size = 2 * r + 1
            K = torch.zeros((size, size), device=device, dtype=dtype)
            deg = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0:
                        continue
                    if neighborhood == 'l1':
                        if abs(dx) + abs(dy) <= r:
                            K[dy + r, dx + r] = 1.0
                            deg += 1
                    elif neighborhood == 'linf':
                        K[dy + r, dx + r] = 1.0
                        deg += 1
            K[r, r] = -deg
            self._kern_cache[key] = K.view(1, 1, size, size)
        return self._kern_cache[key]

    def _lap1d_row(self, W, r, padding):
        _, in_dim = W.shape
        if in_dim <= r:
            return torch.zeros_like(W)
        k = self._get_lap1d_kernel(r, W.device, W.dtype)
        Xp = F.pad(W.unsqueeze(1), (r, r), mode=padding)
        return F.conv1d(Xp, k).squeeze(1)

    def _lap2d_tensor(self, X, r, neighborhood, padding):
        kH, kW = X.size(-2), X.size(-1)
        if kH <= r or kW <= r:
            return torch.zeros_like(X)
        N = X.shape[0] * X.shape[1]
        k = self._get_lap2d_kernel(r, neighborhood, X.device, X.dtype)
        Bp = F.pad(X.view(N, 1, kH, kW), (r, r, r, r), mode=padding)
        return F.conv2d(Bp, k).view_as(X)

    # -----------------------------
    # utilities
    # -----------------------------
    @staticmethod
    def _rms(x, eps=1e-8):
        return x.norm() / (math.sqrt(x.numel()) + eps)

    @staticmethod
    def _softcap_elemwise(X, R, ref_scale, eps=1e-8):
        if R <= 0:
            return X
        cap = R * ref_scale.abs() + 1e-6
        return cap * torch.tanh(X / (cap + eps))

    @staticmethod
    def _split_par_perp(A, u):
        flat_A = A.view(-1)
        flat_u = u.view(-1)
        dot = torch.dot(flat_A, flat_u)
        A_par = dot * u
        A_perp = A - A_par
        return A_par, A_perp

    @staticmethod
    def get_force_scale(current_step, total, ratio):
        if total <= 0:
            return 1.0
        raw_progress = float(current_step) / float(total)
        if ratio < 1e-6:
            eff_progress = 1.0
        else:
            eff_progress = raw_progress / ratio
        eff_progress = max(0.0, min(1.0, eff_progress))
        return 0.5 * (1.0 + math.cos(math.pi * eff_progress))

    @staticmethod
    def _compute_coul_structure_stats_from_Un(Un):
        """
        Un: [K, D] 已归一化向量集合
        返回:
            gram_offdiag_abs_mean
            avg_max_abs_cos
            n_pairs
            n_rows
        """
        out = {
            "gram_offdiag_abs_mean": None,
            "avg_max_abs_cos": None,
            "n_pairs": 0,
            "n_rows": 0,
        }

        if Un is None or Un.shape[0] <= 1:
            return out

        S = torch.mm(Un, Un.t()).abs()  # [K, K]
        eye_mask = torch.eye(Un.shape[0], device=Un.device, dtype=torch.bool)

        offdiag = S[~eye_mask]
        if offdiag.numel() > 0:
            out["gram_offdiag_abs_mean"] = float(offdiag.mean().item())
            out["n_pairs"] = int(offdiag.numel())

        row_max = S.masked_fill(eye_mask, -1.0).max(dim=1).values
        if row_max.numel() > 0:
            out["avg_max_abs_cos"] = float(row_max.mean().item())
            out["n_rows"] = int(row_max.numel())

        return out

    # -----------------------------
    # force construction
    # -----------------------------
    def _compute_force(self, p, g, r_diff, r_coul, enable_diff, enable_coul, cfg, cap_mult):
        """
        Returns:
            F_diff, F_coul, coul_meta
        """
        u = g / (g.norm() + 1e-8)
        info = self.map.get(id(p), {"type": "none", "role": "passive", "name": "UNKNOWN"})
        layer_type = info.get("type", "none")

        F_diff = torch.zeros_like(p)
        F_coul = torch.zeros_like(p)
        coul_meta = {
            "gram_offdiag_abs_mean": None,
            "avg_max_abs_cos": None,
            "n_pairs": 0,
            "n_rows": 0,
        }

        # ---- Diff ----
        if enable_diff:
            raw = torch.zeros_like(p)
            if layer_type == "intra_kernel_conv" and p.dim() == 4:
                raw = self._lap2d_tensor(p, cfg["lap_r"], cfg["lap_neigh"], cfg["lap_padding"])
            elif layer_type == "intra_vector_fc" and p.dim() == 2:
                raw = self._lap1d_row(p, cfg["lap_r"], cfg["lap_padding"])

            if raw.abs().sum() > 0:
                raw = raw / (self._rms(raw) + 1e-8)
                _, raw_perp = self._split_par_perp(raw, u)
                raw_perp = cfg.get("diff_mix_perp", 1.0) * raw_perp

                shaped = self._softcap_elemwise(raw_perp, cap_mult, g)
                F_diff = r_diff * shaped

        # ---- Coul ----
        if enable_coul and info.get("role") == "active":
            U = None
            if p.dim() == 4:
                U = p.view(p.shape[0], -1)
            elif p.dim() == 2:
                U = p

            if U is not None and U.shape[0] > 1:
                Un = F.normalize(U, p=2, dim=1)

                # ---- coul 特色统计 ----
                coul_meta = self._compute_coul_structure_stats_from_Un(Un)

                G = torch.mm(Un, Un.t()) - torch.eye(U.shape[0], device=U.device, dtype=Un.dtype)
                R = -torch.mm(G, Un)
                R = R - (R * Un).sum(dim=1, keepdim=True) * Un
                R = R.view_as(p)

                R = R / (self._rms(R) + 1e-8)
                _, R_perp = self._split_par_perp(R, u)

                shaped = self._softcap_elemwise(R_perp, cap_mult, g)
                F_coul = r_coul * shaped

        return F_diff, F_coul, coul_meta

    # -----------------------------
    # stats API
    # -----------------------------
    def reset_ipd_stats(self):
        # running means over steps
        self._rho_force_sum = 0.0
        self._rho_steps_cnt = 0

        self._rho_diff_sum = 0.0
        self._rho_diff_cnt = 0

        self._rho_coul_sum = 0.0
        self._rho_coul_cnt = 0

        self._cos_Fg_sum = 0.0
        self._cos_Fg_cnt = 0

        self._cos_geff_sum = 0.0
        self._cos_geff_cnt = 0

        self._grad_norm_active_sum = 0.0
        self._grad_norm_active_cnt = 0

        self._force_norm_sum = 0.0
        self._force_norm_cnt = 0

        self._diff_force_norm_sum = 0.0
        self._diff_force_norm_cnt = 0

        self._coul_force_norm_sum = 0.0
        self._coul_force_norm_cnt = 0

        self._gram_offdiag_abs_mean_sum = 0.0
        self._gram_offdiag_abs_mean_cnt = 0

        self._avg_max_abs_cos_sum = 0.0
        self._avg_max_abs_cos_cnt = 0

        self._force_active_steps = 0
        self._force_active_params = 0

        # current step cache
        self._last_step_stats = {
            "global_step": 0,
            "enable_diff": False,
            "enable_coul": False,
            "r_diff_curr": 0.0,
            "r_coul_curr": 0.0,
            "force_active": False,
            "force_active_params": 0,
            "force_active_elems": 0,

            "grad_norm": None,
            "grad_norm_active": None,

            "diff_force_norm": 0.0,
            "coul_force_norm": 0.0,
            "force_norm": 0.0,

            "rho_diff": None,
            "rho_coul": None,
            "rho_force": None,

            "cos_F_vs_-g": None,
            "cos_geff_prev": None,

            "gram_offdiag_abs_mean": None,
            "avg_max_abs_cos": None,
        }

    def get_last_step_stats(self):
        return dict(self._last_step_stats)

    # backward compatibility
    def reset_rho_stats(self):
        self._rho_force_sum = 0.0
        self._rho_steps_cnt = 0
        self._rho_diff_sum = 0.0
        self._rho_diff_cnt = 0
        self._rho_coul_sum = 0.0
        self._rho_coul_cnt = 0

    def reset_traj_stats(self):
        self._cos_Fg_sum = 0.0
        self._cos_Fg_cnt = 0
        self._cos_geff_sum = 0.0
        self._cos_geff_cnt = 0

        self._grad_norm_active_sum = 0.0
        self._grad_norm_active_cnt = 0
        self._force_norm_sum = 0.0
        self._force_norm_cnt = 0
        self._diff_force_norm_sum = 0.0
        self._diff_force_norm_cnt = 0
        self._coul_force_norm_sum = 0.0
        self._coul_force_norm_cnt = 0

        self._gram_offdiag_abs_mean_sum = 0.0
        self._gram_offdiag_abs_mean_cnt = 0
        self._avg_max_abs_cos_sum = 0.0
        self._avg_max_abs_cos_cnt = 0

        self._force_active_steps = 0
        self._force_active_params = 0

    def get_rho_stats(self):
        if self._rho_steps_cnt == 0:
            return None
        return (self._rho_force_sum / self._rho_steps_cnt, self._rho_steps_cnt)

    def get_traj_stats(self):
        out = {}
        if self._cos_Fg_cnt > 0:
            out["cos_F_vs_-g_mean"] = self._cos_Fg_sum / self._cos_Fg_cnt
        if self._cos_geff_cnt > 0:
            out["cos_geff_curv_mean"] = self._cos_geff_sum / self._cos_geff_cnt
        out["n_steps"] = self._rho_steps_cnt
        out["force_active_steps"] = self._force_active_steps
        out["force_active_params"] = self._force_active_params
        return out

    def get_ipd_stats(self):
        rho_force_mean = None
        if self._rho_steps_cnt > 0:
            rho_force_mean = self._rho_force_sum / self._rho_steps_cnt

        rho_diff_mean = None
        if self._rho_diff_cnt > 0:
            rho_diff_mean = self._rho_diff_sum / self._rho_diff_cnt

        rho_coul_mean = None
        if self._rho_coul_cnt > 0:
            rho_coul_mean = self._rho_coul_sum / self._rho_coul_cnt

        cos_Fg_mean = None
        if self._cos_Fg_cnt > 0:
            cos_Fg_mean = self._cos_Fg_sum / self._cos_Fg_cnt

        cos_geff_mean = None
        if self._cos_geff_cnt > 0:
            cos_geff_mean = self._cos_geff_sum / self._cos_geff_cnt

        grad_norm_active_mean = None
        if self._grad_norm_active_cnt > 0:
            grad_norm_active_mean = self._grad_norm_active_sum / self._grad_norm_active_cnt

        force_norm_mean = None
        if self._force_norm_cnt > 0:
            force_norm_mean = self._force_norm_sum / self._force_norm_cnt

        diff_force_norm_mean = None
        if self._diff_force_norm_cnt > 0:
            diff_force_norm_mean = self._diff_force_norm_sum / self._diff_force_norm_cnt

        coul_force_norm_mean = None
        if self._coul_force_norm_cnt > 0:
            coul_force_norm_mean = self._coul_force_norm_sum / self._coul_force_norm_cnt

        gram_offdiag_abs_mean = None
        if self._gram_offdiag_abs_mean_cnt > 0:
            gram_offdiag_abs_mean = self._gram_offdiag_abs_mean_sum / self._gram_offdiag_abs_mean_cnt

        avg_max_abs_cos_mean = None
        if self._avg_max_abs_cos_cnt > 0:
            avg_max_abs_cos_mean = self._avg_max_abs_cos_sum / self._avg_max_abs_cos_cnt

        return {
            "rho_force_mean": rho_force_mean,
            "rho_diff_mean": rho_diff_mean,
            "rho_coul_mean": rho_coul_mean,

            "grad_norm_active_mean": grad_norm_active_mean,
            "force_norm_mean": force_norm_mean,
            "diff_force_norm_mean": diff_force_norm_mean,
            "coul_force_norm_mean": coul_force_norm_mean,

            "gram_offdiag_abs_mean": gram_offdiag_abs_mean,
            "avg_max_abs_cos_mean": avg_max_abs_cos_mean,

            "n_steps": self._rho_steps_cnt,
            "cos_F_vs_-g_mean": cos_Fg_mean,
            "cos_geff_curv_mean": cos_geff_mean,

            "force_active_steps": self._force_active_steps,
            "force_active_params": self._force_active_params,

            "last_step_stats": self.get_last_step_stats(),
        }

    # -----------------------------
    # main step
    # -----------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1
        cfg = self.ipd_config

        scale_diff = self.get_force_scale(
            self.global_step, cfg["total_steps"], cfg["diff_decay_ratio"]
        )
        scale_coul = self.get_force_scale(
            self.global_step, cfg["total_steps"], cfg["coul_decay_ratio"]
        )
        r_diff_curr = cfg["r_diff"] * scale_diff
        r_coul_curr = cfg["r_coul"] * scale_coul

        enable_diff = bool(cfg["enable_diff"]) and (r_diff_curr > 1e-12)
        enable_coul = bool(cfg["enable_coul"]) and (r_coul_curr > 1e-12)

        if self.global_step <= self.debug_first_n_steps:
            print(
                f"[IPD DEBUG] step={self.global_step} "
                f"enable_diff={enable_diff} r_diff_curr={r_diff_curr:.3e} "
                f"enable_coul={enable_coul} r_coul_curr={r_coul_curr:.3e} "
                f"map_size={len(self.map)}"
            )

        self._last_step_stats = {
            "global_step": self.global_step,
            "enable_diff": enable_diff,
            "enable_coul": enable_coul,
            "r_diff_curr": float(r_diff_curr),
            "r_coul_curr": float(r_coul_curr),
            "force_active": False,
            "force_active_params": 0,
            "force_active_elems": 0,

            "grad_norm": None,
            "grad_norm_active": None,

            "diff_force_norm": 0.0,
            "coul_force_norm": 0.0,
            "force_norm": 0.0,

            "rho_diff": None,
            "rho_coul": None,
            "rho_force": None,

            "cos_F_vs_-g": None,
            "cos_geff_prev": None,

            "gram_offdiag_abs_mean": None,
            "avg_max_abs_cos": None,
        }

        if not (enable_diff or enable_coul):
            self.base_optimizer.step()
            return loss

        is_sgd_core = isinstance(self.base_optimizer, optim.SGD) #for post-update
        cap_mult = cfg.get("cap_mult", 1.0)

        post_updates = []

        # active/IPD-support accumulators
        g2_active_sum = 0.0
        fdiff2_sum = 0.0
        fcoul2_sum = 0.0
        ftotal2_sum = 0.0
        n_active = 0

        gram_offdiag_sum = 0.0
        gram_offdiag_cnt = 0

        avg_max_abs_cos_sum = 0.0
        avg_max_abs_cos_cnt = 0

        # cos(F_total, -g)
        dot_F_negG = 0.0
        negG2 = 0.0
        F2 = 0.0

        # cos(geff_t, geff_{t-1})
        dot_geff_prev = 0.0
        geff2 = 0.0
        prev_geff2 = 0.0
        has_prev = False

        step_has_force = False
        step_force_params = 0
        step_force_elems = 0

        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                neg_g = -grad

                F_diff, F_coul, coul_meta = self._compute_force(
                    p=p,
                    g=neg_g,
                    r_diff=r_diff_curr,
                    r_coul=r_coul_curr,
                    enable_diff=enable_diff,
                    enable_coul=enable_coul,
                    cfg=cfg,
                    cap_mult=cap_mult,
                )
                F_total = F_diff + F_coul

                if F_total.abs().sum() == 0:
                    continue

                step_has_force = True
                step_force_params += 1
                step_force_elems += grad.numel()

                # norms on the active/IPD-support set
                g2_active_sum += float((grad * grad).sum().item())
                fdiff2_sum += float((F_diff * F_diff).sum().item())
                fcoul2_sum += float((F_coul * F_coul).sum().item())
                ftotal2_sum += float((F_total * F_total).sum().item())
                n_active += grad.numel()

                # coul structure stats
                v = coul_meta.get("gram_offdiag_abs_mean", None)
                c = coul_meta.get("n_pairs", 0)
                if v is not None and c > 0:
                    gram_offdiag_sum += float(v) * c
                    gram_offdiag_cnt += int(c)

                v = coul_meta.get("avg_max_abs_cos", None)
                c = coul_meta.get("n_rows", 0)
                if v is not None and c > 0:
                    avg_max_abs_cos_sum += float(v) * c
                    avg_max_abs_cos_cnt += int(c)

                # cos(F_total, -g)
                dot_F_negG += float((F_total * neg_g).sum().item())
                negG2 += float((neg_g * neg_g).sum().item())
                F2 += float((F_total * F_total).sum().item())

                # coupled / pre injection
                if is_sgd_core:
                    geff = grad - F_total
                    post_updates.append((p, F_total, lr))
                else:

                    p.grad.add_(F_total, alpha=-1.0)
                    geff = p.grad

                # cos(geff_t, geff_{t-1})
                st = self._get_ipd_state(p)
                prev = st.get("prev_geff", None)
                if prev is not None and prev.shape == geff.shape:
                    dot_geff_prev += float((geff * prev).sum().item())
                    geff2 += float((geff * geff).sum().item())
                    prev_geff2 += float((prev * prev).sum().item())
                    has_prev = True
                st["prev_geff"] = geff.detach().clone()

        if step_has_force:
            self._force_active_steps += 1
            self._force_active_params += step_force_params

        if self.global_step <= self.debug_first_n_steps:
            print(f"[IPD DEBUG] active_force_params={step_force_params}")

        grad_norm_active = None
        diff_force_norm = 0.0
        coul_force_norm = 0.0
        force_norm = 0.0

        rho_diff = None
        rho_coul = None
        rho_force = None

        cos_Fg = None
        cos_geff = None

        gram_offdiag_abs_mean = None
        avg_max_abs_cos = None

        if n_active > 0:
            grad_norm_active = math.sqrt(g2_active_sum)
            diff_force_norm = math.sqrt(fdiff2_sum)
            coul_force_norm = math.sqrt(fcoul2_sum)
            force_norm = math.sqrt(ftotal2_sum)

            grad_rms_active = math.sqrt(g2_active_sum / n_active)
            diff_force_rms = math.sqrt(fdiff2_sum / n_active)
            coul_force_rms = math.sqrt(fcoul2_sum / n_active)
            force_rms = math.sqrt(ftotal2_sum / n_active)

            rho_diff = diff_force_rms / (grad_rms_active + 1e-12)
            rho_coul = coul_force_rms / (grad_rms_active + 1e-12)
            rho_force = force_rms / (grad_rms_active + 1e-12)

            self._rho_force_sum += float(rho_force)
            self._rho_steps_cnt += 1

            self._rho_diff_sum += float(rho_diff)
            self._rho_diff_cnt += 1

            self._rho_coul_sum += float(rho_coul)
            self._rho_coul_cnt += 1

            self._grad_norm_active_sum += float(grad_norm_active)
            self._grad_norm_active_cnt += 1

            self._force_norm_sum += float(force_norm)
            self._force_norm_cnt += 1

            self._diff_force_norm_sum += float(diff_force_norm)
            self._diff_force_norm_cnt += 1

            self._coul_force_norm_sum += float(coul_force_norm)
            self._coul_force_norm_cnt += 1

        if gram_offdiag_cnt > 0:
            gram_offdiag_abs_mean = gram_offdiag_sum / gram_offdiag_cnt
            self._gram_offdiag_abs_mean_sum += float(gram_offdiag_abs_mean)
            self._gram_offdiag_abs_mean_cnt += 1

        if avg_max_abs_cos_cnt > 0:
            avg_max_abs_cos = avg_max_abs_cos_sum / avg_max_abs_cos_cnt
            self._avg_max_abs_cos_sum += float(avg_max_abs_cos)
            self._avg_max_abs_cos_cnt += 1

        if negG2 > 0.0 and F2 > 0.0:
            cos_Fg = dot_F_negG / (math.sqrt(negG2) * math.sqrt(F2) + 1e-12)
            self._cos_Fg_sum += float(cos_Fg)
            self._cos_Fg_cnt += 1

        if has_prev and geff2 > 0.0 and prev_geff2 > 0.0:
            cos_geff = dot_geff_prev / (math.sqrt(geff2) * math.sqrt(prev_geff2) + 1e-12)
            self._cos_geff_sum += float(cos_geff)
            self._cos_geff_cnt += 1

        self._last_step_stats = {
            "global_step": self.global_step,
            "enable_diff": enable_diff,
            "enable_coul": enable_coul,
            "r_diff_curr": float(r_diff_curr),
            "r_coul_curr": float(r_coul_curr),

            "force_active": bool(step_has_force),
            "force_active_params": int(step_force_params),
            "force_active_elems": int(step_force_elems),

            "grad_norm": float(grad_norm_active) if grad_norm_active is not None else None,
            "grad_norm_active": float(grad_norm_active) if grad_norm_active is not None else None,

            "diff_force_norm": float(diff_force_norm),
            "coul_force_norm": float(coul_force_norm),
            "force_norm": float(force_norm),

            "rho_diff": float(rho_diff) if rho_diff is not None else None,
            "rho_coul": float(rho_coul) if rho_coul is not None else None,
            "rho_force": float(rho_force) if rho_force is not None else None,

            "cos_F_vs_-g": float(cos_Fg) if cos_Fg is not None else None,
            "cos_geff_prev": float(cos_geff) if cos_geff is not None else None,

            "gram_offdiag_abs_mean": float(gram_offdiag_abs_mean) if gram_offdiag_abs_mean is not None else None,
            "avg_max_abs_cos": float(avg_max_abs_cos) if avg_max_abs_cos is not None else None,
        }

        self.base_optimizer.step()
        if is_sgd_core and post_updates:
            for p, upd, lr in post_updates:
                if lr != 0.0:
                    p.data.add_(upd, alpha=lr)

        return loss

    # -----------------------------
    # optimizer wrappers
    # -----------------------------
    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "global_step": self.global_step,
            "ipd_state": self.ipd_state,
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict.get("global_step", 0)
        self.ipd_state = state_dict.get("ipd_state", {})
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
