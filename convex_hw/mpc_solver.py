"""
滚动优化 (Model Predictive Control, MPC) 求解带噪声的泊车问题
"""
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds
from typing import Callable, Dict, Optional


class MPCSolver:
    """
    滚动优化求解器
    在每个时间步，根据当前状态求解一个有限时域的最优控制问题，
    只执行第一个控制输入，然后在下一时刻重新优化。
    """

    def __init__(
        self,
        N_horizon: int,  # 预测时域
        x_dim: int,
        u_dim: int,
        dt: float,
        dynamic_f: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
        xN_target: npt.NDArray,
        lower_upper_bound_ux: Dict[str, npt.NDArray],
        noise_std: float = 0.0,  # 状态噪声标准差
    ):
        """
        Parameters
        ----------
        N_horizon : int
            预测时域长度
        x_dim : int
            状态空间维度
        u_dim : int
            控制空间维度
        dt : float
            时间步长
        dynamic_f : Callable
            状态转移函数 dx/dt = f(x, u)
        xN_target : npt.NDArray
            目标终止状态
        lower_upper_bound_ux : Dict
            状态和控制边界
        noise_std : float
            状态噪声标准差
        """
        self.N_horizon = N_horizon
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt = dt
        self.dynamic_f = dynamic_f
        self.xN_target = xN_target
        self.bounds_dict = lower_upper_bound_ux
        self.noise_std = noise_std

    def _simulate_step(self, x: npt.NDArray, u: npt.NDArray, add_noise: bool = True) -> npt.NDArray:
        """模拟一步状态转移（改进欧拉法），可选加噪声"""
        # 使用简单欧拉法进行状态转移（MPC 内部预测）
        f1 = self.dynamic_f(x, u)
        x_pred = x + self.dt * f1

        if add_noise and self.noise_std > 0:
            # 对位置和速度分量加噪声
            noise = np.random.normal(0, self.noise_std, self.x_dim)
            # 对角度分量减小噪声
            noise[3] *= 0.1  # phi (方向盘角度)
            noise[4] *= 0.1  # theta (航向角)
            x_pred = x_pred + noise

        return x_pred

    def _mpc_objective(self, u_seq: npt.NDArray, x_current: npt.NDArray) -> float:
        """
        MPC 目标函数：控制能量 + 终端状态误差
        """
        N = self.N_horizon
        u_seq = u_seq.reshape(N, self.u_dim)

        # 预测轨迹（不加噪声）
        x = x_current.copy()
        cost = 0.0

        # 控制能量项
        Q_u = 1.0  # 控制权重
        for k in range(N):
            cost += Q_u * (u_seq[k, 0]**2 + u_seq[k, 1]**2) * self.dt
            x = self._simulate_step(x, u_seq[k], add_noise=False)

        # 终端状态误差项（增大权重以提高收敛性）
        Q_terminal = np.array([50.0, 50.0, 20.0, 10.0, 30.0])  # 终端状态权重
        terminal_error = x - self.xN_target
        # 对角度误差特殊处理（归一化）
        terminal_error[4] = np.arctan2(np.sin(terminal_error[4]), np.cos(terminal_error[4]))
        cost += np.sum(Q_terminal * terminal_error**2)

        return cost

    def _build_mpc_bounds(self) -> Bounds:
        """构建 MPC 问题的边界约束（只对控制量）"""
        N = self.N_horizon
        lb_u = self.bounds_dict["lb_u"]
        ub_u = self.bounds_dict["ub_u"]

        lower_bounds = []
        upper_bounds = []

        for _ in range(N):
            lower_bounds.extend(lb_u)
            upper_bounds.extend(ub_u)

        return Bounds(lower_bounds, upper_bounds)

    def solve_mpc_step(self, x_current: npt.NDArray, u_prev: Optional[npt.NDArray] = None) -> npt.NDArray:
        """
        求解单步 MPC 问题，返回最优控制序列的第一个控制

        Parameters
        ----------
        x_current : npt.NDArray
            当前状态
        u_prev : Optional[npt.NDArray]
            上一步的控制序列（用于热启动）

        Returns
        -------
        npt.NDArray
            当前时刻的最优控制
        """
        N = self.N_horizon

        # 初始猜测
        if u_prev is not None:
            # 热启动：用上一步的解向前移动
            u_init = np.zeros(N * self.u_dim)
            u_init[:-self.u_dim] = u_prev[self.u_dim:]
            u_init[-self.u_dim:] = u_prev[-self.u_dim:]
        else:
            u_init = np.zeros(N * self.u_dim)

        bounds = self._build_mpc_bounds()

        result = minimize(
            lambda u: self._mpc_objective(u, x_current),
            u_init,
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 100, 'disp': False}
        )

        u_opt = result.x.reshape(N, self.u_dim)
        return u_opt[0], result.x

    def simulate_mpc(self, x0: npt.NDArray, N_total: int) -> tuple:
        """
        使用 MPC 进行完整的轨迹规划

        Parameters
        ----------
        x0 : npt.NDArray
            初始状态
        N_total : int
            总时间步数

        Returns
        -------
        tuple
            (状态轨迹, 控制轨迹)
        """
        x_trajectory = [x0.copy()]
        u_trajectory = []

        x_current = x0.copy()
        u_prev = None

        print("MPC Rolling Optimization with Noise")
        print("=" * 60)
        print(f"Noise std: {self.noise_std}")
        print(f"Prediction horizon: {self.N_horizon}")
        print(f"Total steps: {N_total}")
        print("-" * 60)

        for k in range(N_total):
            # 检查是否已经接近目标
            pos_error = np.linalg.norm(x_current[:2] - self.xN_target[:2])

            if pos_error < 0.1 and abs(x_current[2]) < 0.1:
                print(f"Step {k}: Reached target!")
                # 停止控制
                u_opt = np.array([0.0, 0.0])
            else:
                # 求解 MPC
                u_opt, u_prev = self.solve_mpc_step(x_current, u_prev)

            u_trajectory.append(u_opt.copy())

            # 应用控制（带噪声）
            x_next = self._simulate_step(x_current, u_opt, add_noise=True)

            # 应用状态约束（裁剪）
            x_next[2] = np.clip(x_next[2], self.bounds_dict["lb_x"][2], self.bounds_dict["ub_x"][2])
            x_next[3] = np.clip(x_next[3], self.bounds_dict["lb_x"][3], self.bounds_dict["ub_x"][3])

            x_trajectory.append(x_next.copy())
            x_current = x_next

            if k % 10 == 0:
                print(f"Step {k:3d}: pos=({x_current[0]:.2f}, {x_current[1]:.2f}), "
                      f"v={x_current[2]:.2f}, theta={np.rad2deg(x_current[4]):.1f}°, "
                      f"pos_err={pos_error:.2f}")

        print("-" * 60)
        print(f"Final state: {x_trajectory[-1]}")
        print(f"Target state: {self.xN_target}")

        return np.array(x_trajectory), np.array(u_trajectory)


def run_mpc_parking():
    """运行 MPC 泊车示例"""
    import numpy.typing as npt

    # 问题参数
    T0 = 0.0
    Tf = 20.0
    N_total = 50
    dt = (Tf - T0) / N_total

    X0 = np.array([1.0, 8.0, 0.0, 0.0, 0.0])
    XN = np.array([9.25, 2.0, 0.0, 0.0, np.pi / 2.0])

    LW = 2.8
    V_MAX, V_MIN = 3.0, -2.0
    A_MAX, A_MIN = 2.0, -1.0
    PHI_MAX, OMEGA_MAX = 0.63792, 0.63792

    # 状态转移函数
    def dynamic_f(x: npt.NDArray, u: npt.NDArray) -> npt.NDArray:
        return np.array([
            x[2] * np.cos(x[4]),
            x[2] * np.sin(x[4]),
            u[0],
            u[1],
            x[2] * np.tan(x[3]) / LW
        ], dtype=float)

    lower_upper_bound_ux = {
        "lb_u": np.array((A_MIN, -OMEGA_MAX)),
        "ub_u": np.array((A_MAX, OMEGA_MAX)),
        "lb_x": np.array((-np.inf, -np.inf, V_MIN, -PHI_MAX, -np.inf)),
        "ub_x": np.array((np.inf, np.inf, V_MAX, PHI_MAX, np.inf)),
    }

    # 创建 MPC 求解器（带噪声）
    mpc = MPCSolver(
        N_horizon=25,  # 预测时域（增大以获得更好的收敛性）
        x_dim=5,
        u_dim=2,
        dt=dt,
        dynamic_f=dynamic_f,
        xN_target=XN,
        lower_upper_bound_ux=lower_upper_bound_ux,
        noise_std=0.01,  # 噪声标准差（适度噪声）
    )

    # 设置随机种子以便复现
    np.random.seed(42)

    # 运行 MPC
    x_traj, u_traj = mpc.simulate_mpc(X0, N_total)

    return x_traj, u_traj


if __name__ == "__main__":
    x_traj, u_traj = run_mpc_parking()
    print("\nMPC trajectory shape:", x_traj.shape)
