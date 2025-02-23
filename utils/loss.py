import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad


class LossFunctions(nn.Module):
    def __init__(self, net, X_ic, uv_ic, X_lb, X_ub, X_sample, device, x_min, x_max, t_min, t_max):
        super(LossFunctions, self).__init__()
        self.net = net
        self.device = device
        self.X_ic = X_ic
        self.uv_ic = uv_ic
        self.X_lb = X_lb
        self.X_ub = X_ub
        self.X_sample = X_sample
        self.mse = nn.MSELoss()
        # 将 x_min, x_max, t_min, t_max 保存为类的属性
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max

    def net_uv(self, xt):
        uv = self.net(xt)
        return uv[:, 0:1], uv[:, 1:2]

    def ic_loss(self):
        uv_ic_pred = self.net(self.X_ic)
        u_ic_pred, v_ic_pred = uv_ic_pred[:, 0:1], uv_ic_pred[:, 1:2]
        u_ic, v_ic = self.uv_ic[:, 0:1], self.uv_ic[:, 1:2]
        loss_u_ic = self.mse(u_ic_pred, u_ic)
        loss_v_ic = self.mse(v_ic_pred, v_ic)
        return loss_u_ic, loss_v_ic

    def bc_loss(self):
        X_lb, X_ub = self.X_lb.clone(), self.X_ub.clone()
        X_lb.requires_grad = X_ub.requires_grad = True

        # Dirichlet boundary condition
        u_lb, v_lb = self.net_uv(X_lb)
        u_ub, v_ub = self.net_uv(X_ub)

        mse_bc1_u = self.mse(u_lb, u_ub)
        mse_bc1_v = self.mse(v_lb, v_ub)

        # Neumann boundary condition
        # u_x_lb = grad(u_lb.sum(), X_lb, create_graph=True)[0][:, 0:1]
        # u_x_ub = grad(u_ub.sum(), X_ub, create_graph=True)[0][:, 0:1]
        # v_x_lb = grad(v_lb.sum(), X_lb, create_graph=True)[0][:, 0:1]
        # v_x_ub = grad(v_ub.sum(), X_ub, create_graph=True)[0][:, 0:1]
        #
        # mse_bc2_u = self.loss_fn(u_x_lb, u_x_ub)
        # mse_bc2_v = self.loss_fn(v_x_lb, v_x_ub)

        loss_u_bc = mse_bc1_u  # + mse_bc2_u
        loss_v_bc = mse_bc1_v  # + mse_bc2_v

        return loss_u_bc, loss_v_bc

    def pde_loss(self):
        xt = self.X_sample.clone()
        xt.requires_grad = True
        u, v = self.net_uv(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x, u_t = u_xt[:, 0:1], u_xt[:, 1:2]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        v_xt = grad(v.sum(), xt, create_graph=True)[0]
        v_x, v_t = v_xt[:, 0:1], v_xt[:, 1:2]
        v_xx = grad(v_x.sum(), xt, create_graph=True)[0][:, 0:1]

        f_u = u_t + v_xx + 2 * (u ** 2 + v ** 2) * v
        f_v = v_t - u_xx - 2 * (u ** 2 + v ** 2) * u
        loss_fu = self.mse(f_u, torch.zeros_like(f_u))
        loss_fv = self.mse(f_v, torch.zeros_like(f_v))
        return loss_fu, loss_fv

    def l2_norm_loss(self):
        # 使用传入的 self.x_min, self.x_max, self.t_min, self.t_max
        x_lin = np.linspace(self.x_min, self.x_max, 100)
        t_lin = np.linspace(self.t_min, self.t_max, 100)
        X_mesh, T_mesh = np.meshgrid(x_lin, t_lin)
        q_exact = 2 * np.exp(-2j * X_mesh + 1j) * np.cosh(2 * (X_mesh + 4 * T_mesh)) ** -1
        u_real = np.real(q_exact)
        v_real = np.imag(q_exact)
        X_tensor_local = torch.tensor(X_mesh.flatten(), dtype=torch.float32, device=self.device).unsqueeze(-1)
        T_tensor_local = torch.tensor(T_mesh.flatten(), dtype=torch.float32, device=self.device).unsqueeze(-1)
        q_pred = self.net(torch.cat([X_tensor_local, T_tensor_local], dim=1)).detach().cpu().numpy()
        q_pred = q_pred.reshape(X_mesh.shape[0], X_mesh.shape[1], 2)
        u_pred = q_pred[..., 0]
        v_pred = q_pred[..., 1]
        norm_exact = torch.sqrt(torch.sum(torch.tensor(u_real ** 2 + v_real ** 2, dtype=torch.float32)))
        norm_diff = torch.sqrt(
            torch.sum(torch.tensor((u_pred - u_real) ** 2 + (v_pred - v_real) ** 2, dtype=torch.float32)))
        loss_l2 = norm_diff / norm_exact
        log10_loss_l2 = torch.log10(loss_l2 + 1e-7)
        return loss_l2, log10_loss_l2
