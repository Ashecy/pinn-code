import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_figure(fig, title, directory="figure"):
    """
    Saves the given figure to the specified directory with the provided title.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)  # 创建目录
    filepath = os.path.join(directory, f"{title}.png")
    fig.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {filepath}")


def plotLoss(losses_dict, info=["IC", "BC", "PDE"], filename="training_losses"):
    """
    Plots the training losses for IC, BC, and PDE components and saves the figure.
    """
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), info):
        key = f"loss_{j.lower()}"  # Construct the correct key
        if key in losses_dict:
            axes[i].plot(losses_dict[key])
            axes[i].set_title(j)
        else:
            print(f"Warning: Key '{key}' not found in losses_dict")
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def plot_2d_heatmap(X, T, data, title, colorbar_label, filename="2d_heatmap"):
    """
    Plots a 2D heatmap for the given data and saves it.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.contourf(X, T, data, levels=50, cmap="viridis")
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("t")
    save_figure(fig, filename)
    plt.show()


def plot_3d_surface(X, T, data, title, filename="3d_surface"):
    """
    Plots a 3D surface for the given data and saves it.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, data, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("Predicted Solution Magnitude")
    save_figure(fig, filename)
    plt.show()


def plot_losses(self, filename="loss_components"):
    """
    Plots all loss components except L2 norm losses and saves the figure.
    """
    fig = plt.figure(figsize=(12, 8))

    # Plot loss components excluding L2 losses
    for key in ["loss_u", "loss_v", "loss_fu", "loss_fv"]:
        plt.plot(self.losses[key], label=key)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Components Over Iterations")
    plt.legend()
    plt.grid()
    save_figure(fig, filename)
    plt.show()


def plot_log10_losses(self, filename="log10_loss_components"):
    """
    Plots the log10 of all loss components on the same plot and saves the figure.
    """
    fig = plt.figure(figsize=(12, 8))

    # Plot log10 loss components
    plt.plot(self.losses["log10_loss_u"], label="log10_loss_u")
    plt.plot(self.losses["log10_loss_v"], label="log10_loss_v")
    plt.plot(self.losses["log10_loss_fu"], label="log10_loss_fu")
    plt.plot(self.losses["log10_loss_fv"], label="log10_loss_fv")

    plt.xlabel("Iteration")
    plt.ylabel("log10(Loss)")
    plt.title("Log10 of Loss Components Over Iterations")
    plt.legend()
    plt.grid()
    save_figure(fig, filename)
    plt.show()


def plot_l2_losses(self, filename="l2_losses"):
    """
    Plots the L2 norm loss and its log10 value separately and saves the figure.
    """
    fig = plt.figure(figsize=(12, 6))

    # Plot L2 norm loss
    plt.subplot(1, 2, 1)
    plt.plot(self.losses["loss_l2"], label="L2 Norm Loss", color="b")
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm Loss")
    plt.title("L2 Norm Loss Over Iterations")
    plt.legend()
    plt.grid()

    # Plot log10 of L2 norm loss
    plt.subplot(1, 2, 2)
    plt.plot(self.losses["log10_loss_l2"], label="log10(L2 Norm Loss)", color="r")
    plt.xlabel("Iteration")
    plt.ylabel("log10(L2 Norm Loss)")
    plt.title("Log10(L2 Norm Loss) Over Iterations")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def plot_sampling_points(X_ic, X_lb, X_ub, X_sample, filename="sampling_points"):
    """
    Plots the sampling points for initial condition, boundary condition, and random points and saves the figure.
    """
    fig = plt.figure(figsize=(10, 6))

    # Plot initial condition points
    plt.scatter(X_ic[:, 0].cpu(), X_ic[:, 1].cpu(), color='blue', label='Initial Condition')

    # Plot boundary condition points
    plt.scatter(X_lb[:, 0].cpu(), X_lb[:, 1].cpu(), color='green', label='Boundary Condition (Lower)')
    plt.scatter(X_ub[:, 0].cpu(), X_ub[:, 1].cpu(), color='red', label='Boundary Condition (Upper)')

    # Plot random points
    plt.scatter(X_sample[:, 0].cpu(), X_sample[:, 1].cpu(), color='orange', label='Random Points', alpha=0.1)

    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Sampling Points')
    plt.legend()
    plt.grid()
    save_figure(fig, filename)
    plt.show()



def plot_magnitude_comparison_subplots(pinn, times, x_range=(-5.0, 5.0), filename="magnitude_comparison"):
    """
    Plots the magnitude of the real and predicted solutions |q| at specified times in separate subplots and saves the figure.
    """
    x = np.linspace(*x_range, 500)  # Generate 500 points for x
    fig, axes = plt.subplots(1, len(times), figsize=(24, 8), sharex=True)

    for i, t in enumerate(times):
        # Generate the grid
        X = x
        T = np.full_like(x, t)

        # Compute real solution
        q_exact = 2 * np.exp(-2j * X + 1j) * np.cosh(2 * (X + 4 * T)) ** -1
        q_real = np.sqrt(np.real(q_exact) ** 2 + np.imag(q_exact) ** 2)

        # Compute predicted solution
        X_tensor = torch.tensor(X, dtype=torch.float32, device=pinn.device).unsqueeze(-1)
        T_tensor = torch.tensor(T, dtype=torch.float32, device=pinn.device).unsqueeze(-1)
        q_pred = pinn.net(torch.cat([X_tensor, T_tensor], dim=1)).detach().cpu().numpy()
        q_pred_magnitude = np.sqrt(q_pred[:, 0] ** 2 + q_pred[:, 1] ** 2)

        # Plot real and predicted |q| on the subplot
        axes[i].plot(x, q_real, label=f"Real |q| at t={t}", linestyle="-", color="blue")
        axes[i].plot(x, q_pred_magnitude, label=f"Pred |q| at t={t}", linestyle="--", color="red")

        axes[i].set_title(f"|q| at t = {t}")
        axes[i].set_ylabel("|q|")
        axes[i].legend()
        axes[i].grid()

    # Set common x-label for all subplots
    axes[-1].set_xlabel("x")

    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()
