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
from utils.network import DNN
from utils.monitor import Logger, log_system_info
from utils.loss import LossFunctions
from utils.plotting import Plotter

# 在程序开始时重定向输出
sys.stdout = Logger("output/log.txt")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CUDA acceleration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Load data from .mat file
data = scipy.io.loadmat('data/NLS-one-soliton.mat')

# Extract variables
X = data['X']
T = data['T']
x = data['x']
t = data['t']
u = data['u']
v = data['v']

# change matrix to number
x0 = data['x0'].item()
x1 = data['x1'].item()
t0 = data['t0'].item()
t1 = data['t1'].item()

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


def exact_solution(X, T):
    """
    Computes the exact solution for the given X and T.
    """
    q_exact = 2 * np.exp(-2j * X + 1j) * np.cosh(2 * (X + 4 * T)) ** -1
    u_real, v_real = np.real(q_exact), np.imag(q_exact)
    norm_q_real = np.sqrt(u_real ** 2 + v_real ** 2)
    return u_real, v_real, norm_q_real


# Generate training data
def generate_training_data():
    # x_ic = np.random.uniform(x_min, x_max, (N_ic, 1))
    x_ic = np.random.choice(x.flatten(), (N_ic, 1))
    t_ic = np.full((N_ic, 1), t_min)
    X_ic = np.hstack([x_ic, t_ic])

    q_ic = 2 * np.exp(-2j * x_ic + 1j) * np.cosh(2 * (x_ic - 2)) ** -1
    uv_ic = np.hstack([np.real(q_ic), np.imag(q_ic)])

    t_b = np.random.choice(t.flatten(), (N_bc, 1))
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


class PINN:

    def __init__(self, X_ic, uv_ic, X_lb, X_ub, X_sample, device, start_time):
        self.device = device  # Add device attribute

        # Move data to the specified device
        self.X_ic, self.uv_ic = X_ic.to(device), uv_ic.to(device)
        self.X_lb, self.X_ub, self.X_sample = X_lb.to(device), X_ub.to(device), X_sample.to(device)
        self.lb = lb
        self.ub = ub
        # Initialize the neural network and move it to the device
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
        self.loss_obj = LossFunctions(self.net, self.X_ic, self.uv_ic, self.X_lb, self.X_ub,
                                      self.X_sample, self.device, x_min, x_max, t_min, t_max)
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
        self.start_time = start_time  # 保存开始时间，用于监控

    def net_uv(self, xt):
        uv = self.net(xt)
        return uv[:, 0:1], uv[:, 1:2]

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        loss_u_ic, loss_v_ic = self.loss_obj.ic_loss()
        loss_u_bc, loss_v_bc = self.loss_obj.bc_loss()
        loss_fu, loss_fv = self.loss_obj.pde_loss()
        loss_l2, log10_loss_l2 = self.loss_obj.l2_norm_loss()

        loss_u = 0.5 * loss_u_ic + 0.25 * loss_u_bc
        loss_v = 0.5 * loss_v_ic + 0.25 * loss_v_bc

        total_loss = loss_u + loss_v + loss_fu + loss_fv
        total_loss.backward()

        self.losses["loss_ic"].append((loss_u_ic + loss_v_ic).detach().cpu().item())
        self.losses["loss_bc"].append((loss_u_bc + loss_v_bc).detach().cpu().item())
        self.losses["loss_pde"].append((loss_fu + loss_fv).detach().cpu().item())
        self.losses["log10_loss_ic"].append(torch.log10(loss_u_ic + loss_v_ic + 1e-7).detach().cpu().item())
        self.losses["log10_loss_bc"].append(torch.log10(loss_u_bc + loss_v_bc + 1e-7).detach().cpu().item())
        self.losses["log10_loss_pde"].append(torch.log10(loss_fu + loss_fv + 1e-7).detach().cpu().item())
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
                f"-----------------------------------------------Iteration: {self.iter}-----------------------------------------------")
            print(f"Loss: {total_loss.item():.5e} "
                  f"Loss_u: {loss_u.item():.3e} Loss_v: {loss_v.item():.3e} "
                  f"Loss_fu: {loss_fu.item():.3e} Loss_fv: {loss_fv.item():.3e} "
                  f"L2: {loss_l2.item():.3e}")
            # 使用 monitor.log_system_info 记录监控信息
            log_system_info(self.device, self.start_time, iteration=self.iter)

        return total_loss


if __name__ == "__main__":

    start_time = time.time()  # Record start time

    # Prepare data
    X_ic, uv_ic, X_lb, X_ub, X_sample = generate_training_data()

    # Instantiate the PINN object
    pinn = PINN(X_ic, uv_ic, X_lb, X_ub, X_sample, device, start_time)

    # Adam optimization phase
    for iteration in range(1, 2001):
        pinn.adam.step(pinn.closure)
    print(
        f"=================================================Adam Final=================================================")
    # Log system info after Adam phase
    log_system_info(pinn.device, start_time, label="Adam Final")
    print(f"Adam Optimization Phase: {iteration} iterations completed")

    # LBFGS fine-tuning phase
    pinn.lbfgs.step(pinn.closure)

    print(
        f"=================================================L-BFGS Final=================================================")
    # Log system info after LBFGS phase
    log_system_info(pinn.device, start_time, label="L-BFGS Final")
    print(f"Total Optimization Iterations: {iteration + pinn.iter} iterations completed")

    # Save model
    Path("output").mkdir(parents=True, exist_ok=True)
    torch.save(pinn.net.state_dict(), "output/weight.pt")

    # ============================== plotting ==============================
    # Instantiate Plotter with the losses dictionary from pinn
    plotter = Plotter(losses=pinn.losses)
    # 1. Plot sampling points
    plotter.plot_sampling_points(
        pinn.X_ic, pinn.X_lb, pinn.X_ub, pinn.X_sample, filename="sampling_points"
    )

    # 2. Plot training losses
    plotter.plotLoss(
        pinn.losses, info=["IC", "BC", "PDE"], filename="training_losses"
    )

    # 3. Plot log10 of loss components
    plotter.plot_log10_losses(
        filename="log10_loss_components"
    )

    # 4. Plot L2 norm losses
    plotter.plot_l2_losses(
        filename="l2_losses"
    )

    # Analytical solution
    u_real, v_real, norm_q_real = exact_solution(X, T)

    # Prediction solution
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

    # 5. Plot 2D Heatmap of analytical solution
    plotter.plot_2d_heatmap(
        X,
        T,
        norm_q_real,
        "2D Heatmap of Analytical Solution",
        "Analytical Solution Magnitude",
        filename="heatmap_analytical_solution",
    )

    # 6. Plot 2D Heatmap of predicted solution
    plotter.plot_2d_heatmap(
        X,
        T,
        norm_q_pred,
        "2D Heatmap of Predicted Solution",
        "Predicted Solution Magnitude",
        filename="heatmap_predicted_solution",
    )

    # 7. Plot 2D Heatmap of prediction error
    plotter.plot_2d_heatmap(
        X,
        T,
        error_q,
        "2D Heatmap of Prediction Error",
        "Prediction Error",
        filename="heatmap_prediction_error",
    )

    # 8. Plot 3D Surface plot of predicted solution
    plotter.plot_3d_surface(
        X,
        T,
        norm_q_pred,
        "3D Surface Plot of Predicted Solution",
        filename="3d_predicted_solution",
    )

    # 9. Plot comparisons of |q| at t = -0.25, 0, 0.25
    plotter.plot_magnitude_comparison_subplots(
        pinn, times=[-0.25, 0, 0.25], filename="magnitude_comparison"
    )

    # Save data
    plotter.save_data_to_mat(X, T, norm_q_real, norm_q_pred, error_q)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
