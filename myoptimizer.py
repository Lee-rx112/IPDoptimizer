import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math


class HHDyn(optim.Optimizer):
    def __init__(self, params, lr=1e-2,
                 alpha_leaky=0.1,
                 weight_decay=5e-4,
                 map=None,
                 total_step=10000,
                 force_decay_ratio=0.8,
                 lambda_diff=True,  lambda_coul= False,
                 r_diff = 0.0,
                 r_coul = 0.0,
                 lap_r=1, lap_neigh='l1', lap_padding='reflect',
                 diff_mix_par= 0.2, diff_mix_perp=0.2,
                 max_norm = 10.0
                 ):

        """
                优化器初始化

                Args:
                    params (iterable): 模型参数。
                    lr (float, optional): 主学习率。默认为 1e-3。
                    map (dict, optional): 描述参数邻域关系的图结构。默认为 None。
                    grad_gain: 梯度增益，用于放大梯度信号以匹配动力学尺度。
                    tau (float, optional): 权重更新的放电阈值。默认为 1.2。
                    v_reset (float, optional): 放电后的重置电位。默认为 0.0。
                    lambda_diff (float, optional): 扩散项的强度比例系数。默认为 0.05。
                    lambda_coulomb (float, optional): 库仑力项的强度比例系数。默认为 0.01。
                    dt (float, optional): 动力学方程的离散时间步长。默认为 1.0。
                    quota_scale (float, optional): 方便跨任务调节比例，这里使用默认值1.0
                """

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if any(val > 0 for val in [lambda_diff, lambda_coul]) and map is None:
            raise ValueError("A neighborhood map must be provided if any coupling forces (migration, diffusion, coulomb) are used.")

        defaults = dict(lr=lr,alpha_leaky=alpha_leaky, weight_decay=weight_decay, total_step=total_step,
                        force_decay_ratio=force_decay_ratio,
                        lambda_coul=lambda_coul, r_coul=r_coul,
                        lambda_diff=lambda_diff, r_diff=r_diff,
                        lap_r=lap_r, lap_neigh=lap_neigh, lap_padding=lap_padding,
                        diff_mix_par=diff_mix_par, diff_mix_perp=diff_mix_perp,
                        max_norm=max_norm)
        self.map = map if map is not None else {}

        super(HHDyn, self).__init__(params, defaults)

        self._kern_cache = {}
        # --训练进度（退火用）--
        self._t = 0.0

    @staticmethod
    def get_force_scale(current_step, total, ratio):
        """
        计算解耦的力场缩放因子 (Decoupled Cosine Decay)
        """
        if total <= 0: return 1.0

        raw_progress = float(current_step) / float(total)

        # 计算有效进度
        if ratio < 1e-6:
            eff_progress = 1.0
        else:
            eff_progress = raw_progress / ratio

        # 截断：超过 ratio 后，进度锁定为 1.0 (即力场强度降为 0)
        eff_progress = max(0.0, min(1.0, eff_progress))

        # 余弦衰减: 0.5 * (1 + cos(pi * progress)) -> 从 1.0 降到 0.0
        return 0.5 * (1.0 + math.cos(math.pi * eff_progress))

    @torch.no_grad()
    def cosine_itot_vs_neggrad(self) -> float:
        '''
        诊断：检查总的驱动力与-g的对齐度，检验是否导偏了
        :return:
        '''
        v1_list, v2_list = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p is None:
                    continue
                st = self.state.get(p, None)
                if not st:
                    continue
                itot = st.get('I_total_last', None)
                g = st.get('grad_last', p.grad)
                if itot is None or g is None:
                    continue
                v1_list.append(itot.reshape(-1))
                v2_list.append((-g).reshape(-1))
        if not v1_list:
            return float('nan')
        v1 = torch.cat(v1_list)
        v2 = torch.cat(v2_list)
        denom = (v1.norm() * v2.norm()).clamp_min(1e-12)
        return (v1 @ v2 / denom).item()


    @torch.no_grad()
    def cosine_update_vs_neggrad(self,spiking_only: bool = False) -> float:
        """
            诊断2：cos(dW_last, -grad_last)
            - spiking_only=False（默认）：全量口径，既反映选中谁，也反映方向一致性（更严格，数值通常更低）。
            - spiking_only=True ：仅在放电元素上计算，纯看方向一致性。
        """
        u_list, g_list = [], []
        for group in self.param_groups:
            for p in group['params']:
                st = self.state.get(p, None)
                if not st:
                    continue
                dW = st.get('dW_last', None)
                g = st.get('grad_last', p.grad)
                if dW is None or g is None:
                    continue
                if spiking_only:
                    spk = st.get('spiking_mask', None)
                    if spk is None or not spk.any():  # 该张量本步无人放电
                        continue
                    dW = dW[spk]
                    g = g[spk]
                u_list.append(dW.reshape(-1))
                g_list.append((-g).reshape(-1))
        if not u_list:
            return float('nan')
        u = torch.cat(u_list)
        v = torch.cat(g_list)
        denom = (u.norm() * v.norm()).clamp_min(1e-12)
        return (u @ v / denom).item()

    def get_vw_values(self):
        """
        :return: numpy    Vm in one epoch
        """

        if not self.state:
            return np.array([])

        all_vw = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'membrane_potential' in state:
                    all_vw.append(state['membrane_potential']).detach().cpu().numpy().flatten()
        if not all_vw:
            return np.array([])
        return np.concatenate(all_vw)

    # ------------- 私有工具：标尺/封顶/投影 -------------
    @staticmethod
    def _rms(x, eps=1e-8):
        '''RMS标尺'''
        return x.detach().pow(2).mean().sqrt() + eps

    @staticmethod
    def _softcap_elemwise(X, R, I_grad, eps=1e-8):
        """
        可选：逐元素“软封顶”，把 |X| 限在 ~R*|I_grad| 之内（可导）。
        当前 λ=0 下不会触发邻域力的封顶；后续启用邻域力时会用到。
        """
        cap = R * I_grad.abs()
        return cap * torch.tanh(X / (cap + eps))

    @staticmethod
    def _split_par_perp(A, u, eps=1e-8):
        """
        可选：把 A 沿着 u（单位向量）拆成 ∥ 与 ⟂ 分量。
        启用邻域力且要做“职责分工”时使用；当前 λ=0 不必须。
        """
        # u 需为单位向量（已detach）
        dot = (A.view(-1) * u.view(-1)).sum()
        Apar = (dot * u).view_as(A)
        return Apar, A - Apar

    # ------------- 可选工具：通用拉普拉斯核与卷积：邻域力启用时使用（带缓存） -------------
    def _get_lap1d_kernel(self, r, device, dtype):
        """可选：1D 拉普拉斯核缓存（FC 行内）"""
        key = ('1d', r, device, dtype)
        k = self._kern_cache.get(key)
        if k is None:
            k = torch.zeros(2 * r + 1, device=device, dtype=dtype)
            k[r] = -2 * r
            k[:r] = 1.0
            k[r + 1:] = 1.0
            k = k.view(1, 1, -1)  # [1,1,2r+1]
            self._kern_cache[key] = k
        return k

    def _get_lap2d_kernel(self, r, neighborhood, device, dtype):
        """可选：2D 拉普拉斯核缓存（Conv 核内，'l1' 或 'linf' 邻域）"""
        key = ('2d', r, neighborhood, device, dtype)
        k = self._kern_cache.get(key)
        if k is None:
            size = 2 * r + 1
            K = torch.zeros((size, size), device=device, dtype=dtype)
            deg = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0:
                        continue
                    if neighborhood == 'l1':
                        if abs(dx) + abs(dy) <= r:
                            K[dy + r, dx + r] = 1.0;
                            deg += 1
                    elif neighborhood == 'linf':
                        K[dy + r, dx + r] = 1.0;
                        deg += 1
                    else:
                        raise ValueError("neighborhood must be 'l1' or 'linf'")
            K[r, r] = -deg
            k = K.view(1, 1, size, size)  # [1,1,K,K]
            self._kern_cache[key] = k
        return k

    def _lap1d_row(self, W, r=None, padding=None):
        """FC 行内 1D 拉普拉斯：W [out, in] -> [out, in]"""
        assert W.dim() == 2
        r = self.lap_r if r is None else r
        padding = self.lap_padding if padding is None else padding
        out, in_dim = W.shape
        if in_dim <= r:
            return torch.zeros_like(W)
        k = self._get_lap1d_kernel(r, W.device, W.dtype)  # [1,1,2r+1]
        X = W.unsqueeze(1)  # [out,1,in]
        Xp = F.pad(X, (r, r), mode=padding)
        Y = F.conv1d(Xp, k)  # [out,1,in]
        return Y.squeeze(1)

    def _lap2d_tensor(self, X, r=None, neighborhood=None, padding=None):
        """Conv 核内 2D 拉普拉斯：X [Cout,Cin,kH,kW] -> 同形状"""
        assert X.dim() == 4
        r = self.lap_r if r is None else r
        neighborhood = self.lap_neigh if neighborhood is None else neighborhood
        padding = self.lap_padding if padding is None else padding
        kH, kW = X.size(-2), X.size(-1)
        if kH <= r or kW <= r:
            return torch.zeros_like(X)
        N = X.size(0) * X.size(1)
        B = X.view(N, 1, kH, kW)  # [N,1,H,W]
        k = self._get_lap2d_kernel(r, neighborhood, X.device, X.dtype)  # [1,1,K,K]
        Bp = F.pad(B, (r, r, r, r), mode=padding)
        Y = F.conv2d(Bp, k)
        return Y.view_as(X)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        eps = 1e-8
        for group in self.param_groups:
            # 从group中获取所有超参数
            lr = group['lr']
            alpha_leaky = group['alpha_leaky']
            max_norm = group['max_norm']
            wd = group['weight_decay']
            lambda_diff = group['lambda_diff']
            lambda_coul = group['lambda_coul']
            base_r_diff = group['r_diff']
            r_coul = group['r_coul']
            diff_mix_par = group['diff_mix_par']
            diff_mix_perp = group['diff_mix_perp']
            lap_r = group['lap_r']
            lap_neigh = group['lap_neigh']
            lap_padding = group['lap_padding']
            total_step = group['total_step']
            force_decay_ratio = group.get('force_decay_ratio', 0.8)



            for i_p,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                if max_norm > 0.0:  # 梯度裁剪 (Gradient Clipping)
                    torch.nn.utils.clip_grad_norm_(p, max_norm=max_norm)
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

                # 状态初始化：为每个参数创建一个'potential'张量
                if 'potential' not in state:
                    state['potential'] = torch.zeros_like(p)

                V = state['potential']

                # --- 1. 主电流与标尺 ---
                I_grad =  -grad
                S = self._rms(I_grad) # 以I_grad作为量纲标尺



                # 负梯度的单位方向u，以计算邻域力与 -g 的 ∥/⊥ 分解
                u = -grad
                u = u / (torch.linalg.norm(u) + eps)


                param_id = id(p)
                info = self.map.get(param_id, {'type': 'none', 'role': 'passive'})
                layer_type = info.get('type', 'none')
                current_role = info.get('role', 'passive')
                # ---2. 核内/行内邻域力：迁移和扩散 ---
                D_diff_raw = torch.zeros_like(p)
                if lambda_diff:
                    if layer_type == 'intra_kernel_conv' and p.dim() == 4:
                        D_diff_raw = self._lap2d_tensor(p.detach(), r=lap_r,
                                                                     neighborhood=lap_neigh,
                                                                     padding=lap_padding)

                    elif layer_type == 'intra_vector_fc' and p.dim() == 2:
                        D_diff_raw = self._lap1d_row(p.detach(), r=lap_r,
                                                                  padding=lap_padding)


                I_diffusion= torch.zeros_like(p)


                r_diff = base_r_diff * self.get_force_scale(state['step'], total_step, force_decay_ratio)

                if lambda_diff:
                    I_diff0 = S * (D_diff_raw / (self._rms(D_diff_raw)+eps))
                    I_diff_par, I_diff_perp = self._split_par_perp(I_diff0, u)
                    I_diffusion = (diff_mix_perp * I_diff_perp)+ (diff_mix_par * I_diff_par)
                    I_diffusion = self._softcap_elemwise(I_diffusion, r_diff, I_grad)

                I_coulomb = torch.zeros_like(p)
                if lambda_coul and current_role == 'active':
                    U = None
                    if p.dim() == 4:
                        C_out = p.shape[0]
                        U = p.detach().view(C_out, -1)
                    elif p.dim() == 2:
                        C_out = p.shape[0]
                        U = p.detach()

                    if U is not None and C_out >= 2:
                        Un = U / (U.norm(dim=1, keepdim=True) + eps)  # 每行（通道）L2单位化
                        Gm = Un @ Un.t() # 不同通道的内积，单位化后是余弦相似度
                        # mask = (Gm.abs() > 0.9) & (torch.eye(C_out, device=Gm.device) == 0)
                        # F = - (Gm * mask.float()) @ Un
                        Ieye = torch.eye(C_out, device=U.device, dtype=U.dtype)
                        F = - (Gm - Ieye) @ Un # 自己和自己不需要斥力
                        F = F - (F * Un).sum(dim=1, keepdim=True) * Un # 把F中伸方向的分量去掉，保证值改变方向，不改变模长
                        F = S * (F / (self._rms(F)+eps))  # 规范量纲
                        I_coulomb = F.view_as(p)
                        if r_coul > 0:
                            _, I_cou_perp = self._split_par_perp(I_coulomb, u)
                            I_coulomb = self._softcap_elemwise(I_cou_perp, r_coul, I_grad)


                # 合力、电位更新
                I_total = I_grad  + I_diffusion + I_coulomb
                state['I_total_last'] = I_total.detach()
                state['grad_last'] = grad.detach().clone()

                V.mul_(1.0 - alpha_leaky)  # 上漏
                V.add_(I_total) # 积分

                # --- 3. 权重更新 ---

                with torch.no_grad():
                    dW = lr * V
                    p.data.add_(dW)
                    state['dW_last'] = dW.detach()


        return loss



