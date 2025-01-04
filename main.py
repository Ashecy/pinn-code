import sys

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from pyDOE import lhs

from mpl_toolkits.mplot3d import Axes3D
import time
import psutil
import scipy.io
from utils_plotting import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CUDA acceleration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Load data from .mat file
data = scipy.io.loadmat('data/NLS-soliton.mat')

# Extract variables
X = data['X']
T = data['T']
x0 = data['x0'].item()
x1 = data['x1'].item()
t0= data['t0'].item()
t1 = data['t1'].item()
u = data['u']
v = data['v']

# Compute the magnitude of q
norm_q_real = np.sqrt(u ** 2 + v ** 2)

# Define boundaries
x_min, x_max = x0, x1
t_min, t_max = t0, t1
ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

# Sample sizes
N_ic, N_bc, N_f = 50, 25, 10000

# Convert to torch tensors
X_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
T_tensor = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
norm_q_real_tensor = torch.tensor(norm_q_real, dtype=torch.float32, device=device)


# Generate training data
def generate_training_data():
    x_ic = np.random.uniform(x_min, x_max, (N_ic, 1))
    t_ic = np.full((N_ic, 1), t_min)
    X_ic = np.hstack([x_ic, t_ic])

    u_ic = 2 * np.exp(-2j * x_ic + 1j) * np.cosh(2 * (x_ic - 2)) ** -1
    uv_ic = np.hstack([np.real(u_ic), np.imag(u_ic)])

    t_b = np.random.uniform(t_min, t_max, (N_bc, 1))
    X_lb = np.hstack([np.full((N_bc, 1), x_min), t_b])
    X_ub = np.hstack([np.full((N_bc, 1), x_max), t_b])

    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_sample = np.vstack([X_ic, X_lb, X_ub, X_f])

    # Print the number of sampling points
    print(f"Number of initial condition points: {X_ic.shape[0]}")
    print(f"Number of boundary condition points (lower): {X_lb.shape[0]}")
    print(f"Number of boundary condition points (upper): {X_ub.shape[0]}")
    print(f"Number of total random points: {X_sample.shape[0]}")

    return (
        torch.tensor(X_ic, dtype=torch.float).to(device),
        torch.tensor(uv_ic, dtype=torch.float).to(device),
        torch.tensor(X_lb, dtype=torch.float).to(device),
        torch.tensor(X_ub, dtype=torch.float).to(device),
        torch.tensor(X_sample, dtype=torch.float).to(device),
    )


torch.backends.cuda.matmul.allow_tf32 = (
    False  # This is for Nvidia Ampere GPU Architechture
)

torch.manual_seed(1234)
np.random.seed(1234)


class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  # xavier initialization

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
        out = x
        for layer in self.net:
            out = layer(out)
        return out


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


class PINN:

    def __init__(self, X_ic, uv_ic, X_lb, X_ub, X_sample, device):
        self.device = device  # 添加 device 属性

        # 将数据移动到指定设备
        self.X_ic, self.uv_ic = X_ic.to(device), uv_ic.to(device)
        self.X_lb, self.X_ub, self.X_sample = X_lb.to(device), X_ub.to(device), X_sample.to(device)

        # 初始化神经网络并移动到设备
        self.net = DNN(dim_in=2, dim_out=2, n_layer=9, n_node=40, ub=ub, lb=lb).to(device)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-6,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.losses = {
            "loss_ic": [],
            "loss_bc": [],
            "loss_pde": [],
            "log10_loss_ic": [],
            "log10_loss_bc": [],
            "log10_loss_pde": [],
            "loss_u": [],
            "loss_v": [],
            "loss_fu": [],
            "loss_fv": [],
            "log10_loss_u": [],
            "log10_loss_v": [],
            "log10_loss_fu": [],
            "log10_loss_fv": [],
            "loss_l2": [],
            "log10_loss_l2": []
        }
        self.iter = 0

    def net_uv(self, xt):
        uv = self.net(xt)
        return uv[:, 0:1], uv[:, 1:2]

    def ic_loss(self):
        uv_ic_pred = self.net(self.X_ic)
        u_ic_pred, v_ic_pred = uv_ic_pred[:, 0:1], uv_ic_pred[:, 1:2]
        u_ic, v_ic = self.uv_ic[:, 0:1], self.uv_ic[:, 1:2]
        loss_u = self.loss_fn(u_ic_pred, u_ic)
        loss_v = self.loss_fn(v_ic_pred, v_ic)
        return loss_u, loss_v

    def bc_loss(self):
        X_lb, X_ub = self.X_lb.clone(), self.X_ub.clone()
        X_lb.requires_grad = X_ub.requires_grad = True

        # Dirichlet boundary condition
        u_lb, v_lb = self.net_uv(X_lb)
        u_ub, v_ub = self.net_uv(X_ub)

        mse_bc1_u = self.loss_fn(u_lb, u_ub)
        mse_bc1_v = self.loss_fn(v_lb, v_ub)

        # Neumann boundary condition
        # u_x_lb = grad(u_lb.sum(), X_lb, create_graph=True)[0][:, 0:1]
        # u_x_ub = grad(u_ub.sum(), X_ub, create_graph=True)[0][:, 0:1]
        # v_x_lb = grad(v_lb.sum(), X_lb, create_graph=True)[0][:, 0:1]
        # v_x_ub = grad(v_ub.sum(), X_ub, create_graph=True)[0][:, 0:1]
        #
        # mse_bc2_u = self.loss_fn(u_x_lb, u_x_ub)
        # mse_bc2_v = self.loss_fn(v_x_lb, v_x_ub)

        loss_u = 0.5 * mse_bc1_u  # + mse_bc2_u
        loss_v = 0.5 * mse_bc1_v  # + mse_bc2_v

        return loss_u, loss_v

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

        f_v = v_t - u_xx - 2 * (u ** 2 + v ** 2) * u
        f_u = u_t + v_xx + 2 * (u ** 2 + v ** 2) * v

        f_target = torch.zeros_like(f_u)
        loss_fu = self.loss_fn(f_u, f_target)
        loss_fv = self.loss_fn(f_v, f_target)

        return loss_fu, loss_fv

    def l2_norm_error(self):
        # Generate grid for visualization
        x = np.linspace(x_min, x_max, 100)
        t = np.linspace(t_min, t_max, 100)
        X, T = np.meshgrid(x, t)

        # Analytical solution
        q_exact = 2 * np.exp(-2j * X + 1j) * np.cosh(2 * (X + 4 * T)) ** -1
        u_real, v_real = np.real(q_exact), np.imag(q_exact)

        # Prediction
        X_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
        T_tensor = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
        q_pred = self.net(torch.cat([X_tensor, T_tensor], dim=1)).detach().cpu().numpy().reshape(X.shape[0], X.shape[1],
                                                                                                 2)  # 这里 2 的意思是实部和虚部
        # `q_pred[..., 0]` selects all elements in all dimensions except the last one, and then selects the first element in the last dimension.
        # `q_pred[:, 0]` would select all elements in the first dimension and the first element in the second dimension,
        # which is not the intended operation here.
        u_pred, v_pred = q_pred[..., 0], q_pred[..., 1]

        # Compute L2 norm error
        norm_exact = torch.sqrt(torch.sum(torch.tensor(u_real ** 2 + v_real ** 2, dtype=torch.float32)))
        norm_diff = torch.sqrt(
            torch.sum(torch.tensor((u_pred - u_real) ** 2 + (v_pred - v_real) ** 2, dtype=torch.float32)))
        loss_l2 = norm_diff / norm_exact
        log10_loss_l2 = torch.log10(loss_l2 + 1e-7)

        return loss_l2, log10_loss_l2

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        loss_u_ic, loss_v_ic = self.ic_loss()
        loss_u_bc, loss_v_bc = self.bc_loss()
        loss_fu_pde, loss_fv_pde = self.pde_loss()
        loss_l2, log10_loss_l2 = self.l2_norm_error()

        loss_u = loss_u_ic + loss_u_bc
        loss_v = loss_v_ic + loss_v_bc
        loss_fu = loss_fu_pde
        loss_fv = loss_fv_pde

        total_loss = loss_u + loss_v + loss_fu + loss_fv
        total_loss.backward()

        # Record losses
        self.losses["loss_ic"].append((loss_u_ic + loss_v_ic).detach().cpu().item())
        self.losses["loss_bc"].append((loss_u_bc + loss_v_bc).detach().cpu().item())
        self.losses["loss_pde"].append((loss_fu_pde + loss_fv_pde).detach().cpu().item())
        self.losses["log10_loss_ic"].append(torch.log10(loss_u_ic + loss_v_ic + 1e-7).detach().cpu().item())
        self.losses["log10_loss_bc"].append(torch.log10(loss_u_bc + loss_v_bc + 1e-7).detach().cpu().item())
        self.losses["log10_loss_pde"].append(torch.log10(loss_fu_pde + loss_fv_pde + 1e-7).detach().cpu().item())

        self.losses["loss_u"].append(loss_u.detach().cpu().item())
        self.losses["loss_v"].append(loss_v.detach().cpu().item())
        self.losses["loss_fu"].append(loss_fu.detach().cpu().item())
        self.losses["loss_fv"].append(loss_fv.detach().cpu().item())

        self.losses["log10_loss_u"].append(torch.log10(loss_u + 1e-7).detach().cpu().item())
        self.losses["log10_loss_v"].append(torch.log10(loss_v + 1e-7).detach().cpu().item())
        self.losses["log10_loss_fu"].append(torch.log10(loss_fu + 1e-7).detach().cpu().item())
        self.losses["log10_loss_fv"].append(torch.log10(loss_fv + 1e-7).detach().cpu().item())

        self.losses["loss_l2"].append(loss_l2.detach().cpu().item())
        self.losses["log10_loss_l2"].append(log10_loss_l2.detach().cpu().item())

        self.iter += 1

        if self.iter % 1000 == 0:
            print(
                f"{self.iter}: Loss: {total_loss.item():.5e} "
                f"Loss_u: {loss_u.item():.3e} Loss_v: {loss_v.item():.3e} Loss_fu: {loss_fu.item():.3e} Loss_fv: {loss_fv.item():.3e} "
                f"L2: {loss_l2.item():.3e}"
            )

        return total_loss

    def log_system_info(self, iteration):
        """
        Logs memory usage and GPU memory statistics at specific iterations.
        """
        # Get CPU memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # in MB

        # Get GPU memory usage (if CUDA is available)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
            gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # in MB
            elapsed_time = time.time() - start_time
            print(f"Memory Usage -> CPU: {memory_usage:.2f} MB, "
                  f"GPU Allocated: {gpu_memory_allocated:.2f} MB, GPU Reserved: {gpu_memory_reserved:.2f} MB, "
                  f"Elapsed Time: {elapsed_time:.2f} seconds")
        else:
            print(f"Iteration {iteration}: Memory Usage -> CPU: {memory_usage:.2f} MB")


if __name__ == "__main__":

    start_time = time.time()  # Record start time

    # Prepare data
    X_ic, uv_ic, X_lb, X_ub, X_sample = generate_training_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the PINN object
    pinn = PINN(X_ic, uv_ic, X_lb, X_ub, X_sample, device)

    # Adam optimization phase
    for iteration in range(1, 1001):
        pinn.adam.step(pinn.closure)
        if iteration % 500 == 0:  # Log every 500 iterations
            pinn.log_system_info(iteration)
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration}: Elapsed Time: {elapsed_time:.2f} seconds")

    # Log system info after Adam phase
    pinn.log_system_info("Adam Final")
    print(f"Adam Optimization Phase: {iteration} iterations completed")

    # LBFGS fine-tuning phase
    pinn.lbfgs.step(pinn.closure)

    # Log system info after LBFGS phase
    pinn.log_system_info("LBFGS Final")
    print(f"Total Optimization Iterations: {iteration + pinn.iter} iterations completed")

    # Save model
    Path("output").mkdir(parents=True, exist_ok=True)
    torch.save(pinn.net.state_dict(), "output/weight.pt")

    # ============================== plotting ==============================

    # 1. Plot sampling points
    plot_sampling_points(
        pinn.X_ic, pinn.X_lb, pinn.X_ub, pinn.X_sample, filename="sampling_points"
    )

    # 2. Plot training losses
    plotLoss(
        pinn.losses, info=["IC", "BC", "PDE"], filename="training_losses"
    )

    # 3. Plot log10 of loss components
    plot_log10_losses(
        pinn, filename="log10_loss_components"
    )

    # 4. Plot L2 norm losses
    plot_l2_losses(
        pinn, filename="l2_losses"
    )

    # 5. Generate grid for visualization
    x_min, x_max = -5, 5  # Define x-range
    t_min, t_max = -0.5, 0.5  # Define t-range
    x = np.linspace(x_min, x_max, 100)
    t = np.linspace(t_min, t_max, 100)
    X, T = np.meshgrid(x, t)

    # 6. Analytical solution
    q_exact = 2 * np.exp(-2j * X + 1j) * np.cosh(2 * (X + 4 * T)) ** -1
    u_real, v_real = np.real(q_exact), np.imag(q_exact)
    norm_q_real = np.sqrt(u_real ** 2 + v_real ** 2)

    # 7. Prediction
    X_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    T_tensor = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(-1)
    q_pred = (
        pinn.net(torch.cat([X_tensor, T_tensor], dim=1))
        .detach()
        .cpu()
        .numpy()
        .reshape(X.shape[0], X.shape[1], 2)
    )
    u_pred, v_pred = q_pred[..., 0], q_pred[..., 1]
    norm_q_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    error_q = norm_q_real - norm_q_pred

    # 8. 2D Heatmap of analytical solution
    plot_2d_heatmap(
        X,
        T,
        norm_q_real,
        "2D Heatmap of Analytical Solution",
        "Analytical Solution Magnitude",
        filename="heatmap_analytical_solution",
    )

    # 9. 2D Heatmap of predicted solution
    plot_2d_heatmap(
        X,
        T,
        norm_q_pred,
        "2D Heatmap of Predicted Solution",
        "Predicted Solution Magnitude",
        filename="heatmap_predicted_solution",
    )

    # 10. 2D Heatmap of prediction error
    plot_2d_heatmap(
        X,
        T,
        error_q,
        "2D Heatmap of Prediction Error",
        "Prediction Error",
        filename="heatmap_prediction_error",
    )

    # 11. 3D Surface plot of predicted solution
    plot_3d_surface(
        X,
        T,
        norm_q_pred,
        "3D Surface Plot of Predicted Solution",
        filename="3d_predicted_solution",
    )

    # 12. Plot comparisons of |q| at t = -0.25, 0, 0.25
    plot_magnitude_comparison_subplots(
        pinn, times=[-0.25, 0, 0.25], filename="magnitude_comparison"
    )

    # Call the function to save data
    save_data_to_mat(X, T, norm_q_real, norm_q_pred, error_q)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
