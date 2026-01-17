"""
PID 控制器跟踪参考轨迹
利用带噪声的状态方程，采用 PID 跟踪泊车轨迹
"""
import numpy as np
import numpy.typing as npt
from typing import Optional


class PIDController:
    """PID 控制器"""

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        output_limits: tuple = (-np.inf, np.inf),
        dt: float = 0.1
    ):
        """
        Parameters
        ----------
        Kp : float
            比例增益
        Ki : float
            积分增益
        Kd : float
            微分增益
        output_limits : tuple
            输出限幅
        dt : float
            采样时间
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float) -> float:
        """
        计算 PID 控制输出

        Parameters
        ----------
        error : float
            当前误差

        Returns
        -------
        float
            控制输出
        """
        # 比例项
        P = self.Kp * error

        # 积分项（带抗饱和）
        self.integral += error * self.dt
        # 积分限幅
        integral_limit = 10.0
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)
        I = self.Ki * self.integral

        # 微分项
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative

        self.prev_error = error

        # 总输出
        output = P + I + D

        # 输出限幅
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output


class VehiclePIDTracker:
    """
    车辆 PID 轨迹跟踪器
    使用两个独立的 PID 控制器分别控制纵向（加速度）和横向（方向盘角速度）
    """

    def __init__(
        self,
        dt: float,
        Lw: float,
        bounds_dict: dict,
        noise_std: float = 0.0
    ):
        """
        Parameters
        ----------
        dt : float
            时间步长
        Lw : float
            车辆轴距
        bounds_dict : dict
            控制和状态边界
        noise_std : float
            状态噪声标准差
        """
        self.dt = dt
        self.Lw = Lw
        self.bounds_dict = bounds_dict
        self.noise_std = noise_std

        # 纵向 PID（控制加速度以跟踪速度）
        self.pid_longitudinal = PIDController(
            Kp=2.0, Ki=0.5, Kd=0.1,
            output_limits=(bounds_dict["lb_u"][0], bounds_dict["ub_u"][0]),
            dt=dt
        )

        # 横向 PID（控制方向盘角速度以跟踪航向角）
        self.pid_lateral = PIDController(
            Kp=1.5, Ki=0.3, Kd=0.2,
            output_limits=(bounds_dict["lb_u"][1], bounds_dict["ub_u"][1]),
            dt=dt
        )

        # 位置跟踪 PID（用于生成期望速度）
        self.pid_position = PIDController(
            Kp=0.8, Ki=0.1, Kd=0.1,
            output_limits=(-2.0, 3.0),  # 速度范围
            dt=dt
        )

    def dynamic_f(self, x: npt.NDArray, u: npt.NDArray) -> npt.NDArray:
        """状态转移函数"""
        return np.array([
            x[2] * np.cos(x[4]),
            x[2] * np.sin(x[4]),
            u[0],
            u[1],
            x[2] * np.tan(x[3]) / self.Lw
        ], dtype=float)

    def simulate_step(self, x: npt.NDArray, u: npt.NDArray, add_noise: bool = True) -> npt.NDArray:
        """模拟一步状态转移（带噪声）"""
        f1 = self.dynamic_f(x, u)
        x_next = x + self.dt * f1

        if add_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, len(x))
            noise[3] *= 0.1  # 减小角度分量噪声
            noise[4] *= 0.1
            x_next = x_next + noise

        # 应用状态约束
        x_next[2] = np.clip(x_next[2], self.bounds_dict["lb_x"][2], self.bounds_dict["ub_x"][2])
        x_next[3] = np.clip(x_next[3], self.bounds_dict["lb_x"][3], self.bounds_dict["ub_x"][3])

        return x_next

    def compute_control(
        self,
        x_current: npt.NDArray,
        x_ref: npt.NDArray,
        v_ref: float,
        theta_ref: float
    ) -> npt.NDArray:
        """
        计算 PID 控制输入

        Parameters
        ----------
        x_current : npt.NDArray
            当前状态
        x_ref : npt.NDArray
            参考状态
        v_ref : float
            参考速度
        theta_ref : float
            参考航向角

        Returns
        -------
        npt.NDArray
            控制输入 [a, omega]
        """
        # 位置误差（用于调整速度）
        pos_error = np.sqrt((x_ref[0] - x_current[0])**2 + (x_ref[1] - x_current[1])**2)

        # 计算期望航向角（指向参考点）
        dx = x_ref[0] - x_current[0]
        dy = x_ref[1] - x_current[1]
        if pos_error > 0.1:
            desired_theta = np.arctan2(dy, dx)
        else:
            desired_theta = theta_ref

        # 航向角误差（归一化到 [-pi, pi]）
        theta_error = desired_theta - x_current[4]
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

        # 速度误差
        # 如果偏离轨迹太多或航向角误差大，减小速度
        if abs(theta_error) > np.pi / 4:
            v_desired = 0.3 * v_ref
        else:
            v_desired = v_ref

        v_error = v_desired - x_current[2]

        # 计算控制输入
        a = self.pid_longitudinal.compute(v_error)
        omega = self.pid_lateral.compute(theta_error)

        return np.array([a, omega])

    def track_trajectory(
        self,
        x0: npt.NDArray,
        ref_trajectory: npt.NDArray,
        ref_controls: Optional[npt.NDArray] = None
    ) -> tuple:
        """
        跟踪参考轨迹

        Parameters
        ----------
        x0 : npt.NDArray
            初始状态
        ref_trajectory : npt.NDArray
            参考轨迹 shape=(N+1, x_dim)
        ref_controls : Optional[npt.NDArray]
            参考控制 shape=(N, u_dim)

        Returns
        -------
        tuple
            (实际轨迹, 实际控制, 跟踪误差)
        """
        N = len(ref_trajectory) - 1

        actual_trajectory = [x0.copy()]
        actual_controls = []
        tracking_errors = []

        x_current = x0.copy()

        # 重置 PID 控制器
        self.pid_longitudinal.reset()
        self.pid_lateral.reset()
        self.pid_position.reset()

        print("PID Trajectory Tracking with Noise")
        print("=" * 60)
        print(f"Noise std: {self.noise_std}")
        print(f"Trajectory length: {N + 1}")
        print("-" * 60)

        for k in range(N):
            x_ref = ref_trajectory[k + 1]  # 下一个参考点
            v_ref = ref_trajectory[k + 1, 2] if k + 1 < len(ref_trajectory) else 0.0
            theta_ref = ref_trajectory[k + 1, 4] if k + 1 < len(ref_trajectory) else ref_trajectory[-1, 4]

            # 计算 PID 控制
            u = self.compute_control(x_current, x_ref, v_ref, theta_ref)

            # 应用控制并模拟（带噪声）
            x_next = self.simulate_step(x_current, u, add_noise=True)

            # 记录
            actual_trajectory.append(x_next.copy())
            actual_controls.append(u.copy())

            # 计算跟踪误差
            pos_error = np.sqrt((x_next[0] - x_ref[0])**2 + (x_next[1] - x_ref[1])**2)
            tracking_errors.append(pos_error)

            x_current = x_next

            if k % 10 == 0:
                print(f"Step {k:3d}: pos=({x_current[0]:.2f}, {x_current[1]:.2f}), "
                      f"ref=({x_ref[0]:.2f}, {x_ref[1]:.2f}), "
                      f"error={pos_error:.3f}")

        print("-" * 60)
        print(f"Final state: {actual_trajectory[-1]}")
        print(f"Target state: {ref_trajectory[-1]}")
        print(f"Average tracking error: {np.mean(tracking_errors):.4f}")
        print(f"Max tracking error: {np.max(tracking_errors):.4f}")

        return np.array(actual_trajectory), np.array(actual_controls), np.array(tracking_errors)


def run_pid_tracking():
    """运行 PID 跟踪示例"""
    from opt_control import OptControl

    # 问题参数
    T0 = 0.0
    Tf = 20.0
    N = 50
    H = (Tf - T0) / N

    X0 = np.array([1.0, 8.0, 0.0, 0.0, 0.0])
    XN = np.array([9.25, 2.0, 0.0, 0.0, np.pi / 2.0])

    LW = 2.8
    V_MAX, V_MIN = 3.0, -2.0
    A_MAX, A_MIN = 2.0, -1.0
    PHI_MAX, OMEGA_MAX = 0.63792, 0.63792

    lower_upper_bound_ux = {
        "lb_u": np.array((A_MIN, -OMEGA_MAX)),
        "ub_u": np.array((A_MAX, OMEGA_MAX)),
        "lb_x": np.array((-np.inf, -np.inf, V_MIN, -PHI_MAX, -np.inf)),
        "ub_x": np.array((np.inf, np.inf, V_MAX, PHI_MAX, np.inf)),
    }

    # 首先生成参考轨迹（使用原始优化求解器）
    def J(ux):
        u1 = ux[0 : N + 1]
        u2 = ux[N + 1 : 2 * (N + 1)]
        res = 0.0
        for i in range(0, N):
            res += (u1[i] ** 2 + u1[i + 1] ** 2) * H / 2.0
            res += (u2[i] ** 2 + u2[i + 1] ** 2) * H / 2.0
        return res

    def dynamic_f_gen(Lw: float = 3.0):
        def dynamic_f(x: npt.NDArray, u: npt.NDArray) -> npt.NDArray:
            return np.array([
                x[2] * np.cos(x[4]),
                x[2] * np.sin(x[4]),
                u[0],
                u[1],
                x[2] * np.tan(x[3]) / Lw
            ], dtype=float)
        return dynamic_f

    dynamic_f = dynamic_f_gen(LW)

    def dyn_cons(xk, xkp1, uk, ukp1):
        return xkp1 - xk - (dynamic_f(xk, uk) + dynamic_f(xkp1, ukp1)) * H / 2.0

    print("Step 1: Generating reference trajectory...")
    opt = OptControl(
        N=N, x_dim=5, u_dim=2, J=J,
        dyn_cons=dyn_cons, x0=X0, xN=XN,
        lower_upper_bound_ux=lower_upper_bound_ux,
    )
    x0_guess = 0.01 * np.ones((N + 1) * 7)
    ref_trajectory, ref_controls = opt.solve(init_guess=x0_guess)

    print("\nStep 2: PID tracking with noise...")
    np.random.seed(42)

    tracker = VehiclePIDTracker(
        dt=H,
        Lw=LW,
        bounds_dict=lower_upper_bound_ux,
        noise_std=0.02
    )

    actual_traj, actual_ctrl, errors = tracker.track_trajectory(
        X0, ref_trajectory, ref_controls
    )

    return ref_trajectory, actual_traj, actual_ctrl, errors


if __name__ == "__main__":
    ref_traj, actual_traj, actual_ctrl, errors = run_pid_tracking()
    print("\nPID tracking completed!")
    print(f"Reference trajectory shape: {ref_traj.shape}")
    print(f"Actual trajectory shape: {actual_traj.shape}")
