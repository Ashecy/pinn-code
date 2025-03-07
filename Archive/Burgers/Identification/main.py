import sys

sys.path.append("..")
import numpy as np
import torch
from torch.autograd import grad
from network import DNN
from scipy.io import loadmat

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


"""
Burgers Eqn.
反问题
f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx = 0, x ~ [-1, 1], t ~ [0, 1]
lambda_1 = 1
lambda_2 = 0.01/pi = 0.0031831
"""
N_u = 2000

data = loadmat("../burgers_shock.mat")
x = data["x"]
t = data["t"]
u = data["usol"].T

ub = np.array([x.max(), t.max()])
lb = np.array([x.min(), t.min()])

# Clean Data Preparation
x_, t_ = np.meshgrid(x, t)
x_ = x_.reshape(-1, 1)
t_ = t_.reshape(-1, 1)
u_ = u.reshape(-1, 1)

rand_idx = np.random.choice(len(u_), N_u, replace=False)

x = torch.tensor(x_[rand_idx], dtype=torch.float32).to(device)
t = torch.tensor(t_[rand_idx], dtype=torch.float32).to(device)
xt = torch.cat((x, t), dim=1)
u = torch.tensor(u_[rand_idx], dtype=torch.float32).to(device)

# 1% Noisy Data Preparation
noise = 0.01
noisy_u = u_ + noise * np.std(u_) * np.random.randn(*u_.shape)
noisy_u = torch.tensor(noisy_u[rand_idx], dtype=torch.float32).to(device)


class PINN:
    def __init__(self, u):
        self.u = u
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.net = DNN(dim_in=2, dim_out=1, n_layer=7, n_node=20, ub=ub, lb=lb,).to(
            device
        )
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)

        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def f(self, xt):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]

        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def closure(self):
        self.optimizer.zero_grad()

        u_pred = self.net(xt)
        f_pred = self.f(xt)

        mse_u = torch.mean(torch.square(u_pred - self.u))
        mse_f = torch.mean(torch.square(f_pred))

        loss = mse_u + mse_f
        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.lambda_1.item():.5f}, l2 : {torch.exp(self.lambda_2).item():.5f}",
            end="",
        )
        """
        这是因为代码使用了 `\r` 字符来在同一行上动态更新输出，而在每 500 次迭代时，使用 `print("")` 来换行并输出新的信息。
        如果你想修改这个行为，可以调整代码中的 `print` 语句。
        例如，如果你想每次都输出新的行而不是动态更新，可以删除 `\r` 和 `end=""` 参数。
        """
        if self.iter % 500 == 0:
            print("")
        return loss


def calcError(pinn):
    u_pred = pinn.net(torch.hstack((x, t)))
    u_pred = u_pred.detach().cpu().numpy()
    u_ = u.detach().cpu().numpy()
    error_u = np.linalg.norm(u_ - u_pred, 2) / np.linalg.norm(u_, 2)
    lambda1 = pinn.lambda_1.detach().cpu().item()
    lambda2 = np.exp(pinn.lambda_2.detach().cpu().item())
    error_lambda1 = np.abs(lambda1 - 1.0) * 100
    error_lambda2 = np.abs(lambda2 - 0.01 / np.pi) * 100
    print(
        f"\nError u  : {error_u:.5e}",
        f"\nError l1 : {error_lambda1:.5f}%",
        f"\nError l2 : {error_lambda2:.5f}%",
    )
    return (error_u, error_lambda1, error_lambda2)


if __name__ == "__main__":
    pinn = PINN(u)
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "weight_clean.pt")
    pinn.net.load_state_dict(torch.load("weight_clean.pt"))
    calcError(pinn)

    pinn = PINN(noisy_u)
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "weight_noisy1.pt")
    calcError(pinn)
