import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from opt_control import OptControl
from mpc_solver import MPCSolver
from pid_tracker import VehiclePIDTracker

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

def dynamic_f(x: npt.NDArray, u: npt.NDArray) -> npt.NDArray:
    return np.array([
        x[2] * np.cos(x[4]),
        x[2] * np.sin(x[4]),
        u[0],
        u[1],
        x[2] * np.tan(x[3]) / LW
    ], dtype=float)

def J(ux):
    u1 = ux[0 : N + 1]
    u2 = ux[N + 1 : 2 * (N + 1)]
    res = 0.0
    for i in range(0, N):
        res += (u1[i] ** 2 + u1[i + 1] ** 2) * H / 2.0
        res += (u2[i] ** 2 + u2[i + 1] ** 2) * H / 2.0
    return res

def dyn_cons(xk, xkp1, uk, ukp1):
    return xkp1 - xk - (dynamic_f(xk, uk) + dynamic_f(xkp1, ukp1)) * H / 2.0


print("=" * 70)
print("基础求解器: 使用 scipy")
print("=" * 70)
opt = OptControl(
    N=N, x_dim=5, u_dim=2, J=J,
    dyn_cons=dyn_cons, x0=X0, xN=XN,
    lower_upper_bound_ux=lower_upper_bound_ux,
)
x0_guess = 0.01 * np.ones((N + 1) * 7)
ref_trajectory, ref_controls = opt.solve(init_guess=x0_guess)

print("\n参考轨迹生成完成!")
print(f"初始状态: {ref_trajectory[0]}")
print(f"末端状态: {ref_trajectory[-1]}")


print("\n" + "=" * 70)
print("MPC 滚动优化: 带噪声")
print("=" * 70)
np.random.seed(42)

mpc = MPCSolver(
    N_horizon=25,
    x_dim=5,
    u_dim=2,
    dt=H,
    dynamic_f=dynamic_f,
    xN_target=XN,
    lower_upper_bound_ux=lower_upper_bound_ux,
    noise_std=0.01,
)
mpc_trajectory, mpc_controls = mpc.simulate_mpc(X0, N)


print("\n" + "=" * 70)
print("PID 轨迹跟踪: 带噪声")
print("=" * 70)
np.random.seed(42)

tracker = VehiclePIDTracker(
    dt=H,
    Lw=LW,
    bounds_dict=lower_upper_bound_ux,
    noise_std=0.01  # 噪声标准差
)
pid_trajectory, pid_controls, tracking_errors = tracker.track_trajectory(
    X0, ref_trajectory, ref_controls
)


# 可视化结果
print("\n" + "=" * 70)
print("生成可视化结果...")
print("=" * 70)

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# 1. 轨迹对比图
ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], 'b-', linewidth=2, label='Reference (Optimal)')
ax.plot(mpc_trajectory[:, 0], mpc_trajectory[:, 1], 'r--', linewidth=2, label='MPC with Noise')
ax.plot(pid_trajectory[:, 0], pid_trajectory[:, 1], 'g-.', linewidth=2, label='PID Tracking')
ax.plot(X0[0], X0[1], 'ko', markersize=10, label='Start')
ax.plot(XN[0], XN[1], 'k*', markersize=15, label='Target')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Trajectory Comparison')
ax.legend()
ax.grid(True)
ax.axis('equal')

plt.tight_layout()
plt.savefig('/home/shang_sun/convex_hw/bonus_results.png', dpi=150)
print("可视化结果已保存到 bonus_results.png")

# 打印总结
print("\n" + "=" * 70)
print("结果总结")
print("=" * 70)
print(f"\n1. 参考轨迹: 最优控制")
print(f"   - 初始位置: ({ref_trajectory[0, 0]:.2f}, {ref_trajectory[0, 1]:.2f})")
print(f"   - 末端位置: ({ref_trajectory[-1, 0]:.2f}, {ref_trajectory[-1, 1]:.2f})")
print(f"   - 末端航向角: {np.rad2deg(ref_trajectory[-1, 4]):.1f}°")

print(f"\n2. MPC 轨迹: 带噪声滚动优化")
print(f"   - 初始位置: ({mpc_trajectory[0, 0]:.2f}, {mpc_trajectory[0, 1]:.2f})")
print(f"   - 末端位置: ({mpc_trajectory[-1, 0]:.2f}, {mpc_trajectory[-1, 1]:.2f})")
print(f"   - 末端航向角: {np.rad2deg(mpc_trajectory[-1, 4]):.1f}°")
print(f"   - 与目标位置误差: {np.linalg.norm(mpc_trajectory[-1, :2] - XN[:2]):.3f}m")

print(f"\n3. PID 跟踪轨迹: 带噪声")
print(f"   - 初始位置: ({pid_trajectory[0, 0]:.2f}, {pid_trajectory[0, 1]:.2f})")
print(f"   - 末端位置: ({pid_trajectory[-1, 0]:.2f}, {pid_trajectory[-1, 1]:.2f})")
print(f"   - 末端航向角: {np.rad2deg(pid_trajectory[-1, 4]):.1f}°")
print(f"   - 平均跟踪误差: {np.mean(tracking_errors):.3f}m")
print(f"   - 最大跟踪误差: {np.max(tracking_errors):.3f}m")

# 输出 MPC 完整轨迹
print("\n" + "=" * 70)
print("MPC 轨迹 [x0, x1, ..., xN] (每行为 [px, py, v, phi, theta]):")
print("=" * 70)
for i in range(len(mpc_trajectory)):
    print(f"x[{i:2d}] = [{mpc_trajectory[i, 0]:8.4f}, {mpc_trajectory[i, 1]:8.4f}, "
          f"{mpc_trajectory[i, 2]:8.4f}, {mpc_trajectory[i, 3]:8.4f}, {mpc_trajectory[i, 4]:8.4f}]")
