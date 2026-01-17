"""
最优控制问题求解器
使用 scipy.optimize.minimize 的 SLSQP 方法求解离散化后的最优控制问题
"""
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds
from typing import Callable, Dict, Optional


class OptControl:
    """最优控制问题求解器

    求解如下形式的最优控制问题:
    min_{x_k, u_k} J(u)
    s.t. x_{k+1} = x_k + dt/2 * [f(x_k, u_k) + f(x_{k+1}, u_{k+1})]  (状态转移约束)
         lb_u <= u_k <= ub_u  (控制量约束)
         lb_x <= x_k <= ub_x  (状态量约束)
         x_0 = x0  (初始状态约束)
         x_N = xN  (末端状态约束)
    """

    def __init__(
        self,
        N: int,
        x_dim: int,
        u_dim: int,
        J: Callable[[npt.NDArray], float],
        dyn_cons: Callable[[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray],
        x0: npt.NDArray,
        xN: npt.NDArray,
        lower_upper_bound_ux: Dict[str, npt.NDArray],
    ):
        """
        Parameters
        ----------
        N : int
            时间离散化的区间数
        x_dim : int
            状态空间维度
        u_dim : int
            控制空间维度
        J : Callable
            目标函数，输入为优化变量向量，输出为目标值
        dyn_cons : Callable
            状态转移约束函数，输入为 (x_k, x_{k+1}, u_k, u_{k+1})，输出应为 0
        x0 : npt.NDArray
            初始状态约束
        xN : npt.NDArray
            末端状态约束
        lower_upper_bound_ux : Dict
            包含 lb_u, ub_u, lb_x, ub_x 的字典
        """
        self.N = N
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.J = J
        self.dyn_cons = dyn_cons
        self.x0 = x0
        self.xN = xN
        self.bounds_dict = lower_upper_bound_ux

        self.n_vars = (N + 1) * (u_dim + x_dim)

    def _extract_u_x(self, z: npt.NDArray) -> tuple:
        """从优化变量向量中提取 u 和 x 序列

        Parameters
        ----------
        z : npt.NDArray
            优化变量向量

        Returns
        -------
        tuple
            (u_array, x_array) 其中 u_array 形状为 (N+1, u_dim), x_array 形状为 (N+1, x_dim)
        """
        N = self.N
        u_dim = self.u_dim
        x_dim = self.x_dim

        # u 存储: u1[0:N+1], u2[0:N+1]
        u_array = np.zeros((N + 1, u_dim))
        for i in range(u_dim):
            u_array[:, i] = z[i * (N + 1): (i + 1) * (N + 1)]

        # x 存储: x1[0:N+1], x2[0:N+1], ..., x5[0:N+1]
        x_start = u_dim * (N + 1)
        x_array = np.zeros((N + 1, x_dim))
        for i in range(x_dim):
            x_array[:, i] = z[x_start + i * (N + 1): x_start + (i + 1) * (N + 1)]

        return u_array, x_array

    def _build_bounds(self) -> Bounds:
        """构建优化变量的边界约束"""
        N = self.N
        u_dim = self.u_dim
        x_dim = self.x_dim

        lb_u = self.bounds_dict["lb_u"]
        ub_u = self.bounds_dict["ub_u"]
        lb_x = self.bounds_dict["lb_x"]
        ub_x = self.bounds_dict["ub_x"]

        lower_bounds = []
        upper_bounds = []

        # u 的边界
        for i in range(u_dim):
            lower_bounds.extend([lb_u[i]] * (N + 1))
            upper_bounds.extend([ub_u[i]] * (N + 1))

        # x 的边界
        for i in range(x_dim):
            lower_bounds.extend([lb_x[i]] * (N + 1))
            upper_bounds.extend([ub_x[i]] * (N + 1))

        return Bounds(lower_bounds, upper_bounds)

    def _build_constraints(self):
        """构建等式约束：状态转移约束、初始约束、末端约束"""
        N = self.N
        x_dim = self.x_dim
        u_dim = self.u_dim

        constraints = []

        # 1. 状态转移约束 (等式约束)
        # x_{k+1} - x_k - dt/2 * [f(x_k, u_k) + f(x_{k+1}, u_{k+1})] = 0
        def dynamics_constraint(z):
            u_array, x_array = self._extract_u_x(z)
            residuals = []
            for k in range(N):
                xk = x_array[k]
                xkp1 = x_array[k + 1]
                uk = u_array[k]
                ukp1 = u_array[k + 1]
                res = self.dyn_cons(xk, xkp1, uk, ukp1)
                residuals.extend(res)
            return np.array(residuals)

        constraints.append({
            'type': 'eq',
            'fun': dynamics_constraint
        })

        # 2. 初始状态约束 x_0 = x0
        def initial_constraint(z):
            _, x_array = self._extract_u_x(z)
            return x_array[0] - self.x0

        constraints.append({
            'type': 'eq',
            'fun': initial_constraint
        })

        # 3. 末端状态约束 x_N = xN
        def terminal_state_constraint(z):
            _, x_array = self._extract_u_x(z)
            return x_array[N] - self.xN

        constraints.append({
            'type': 'eq',
            'fun': terminal_state_constraint
        })

        # 4. 末端控制约束 u_N = [0, 0]
        def terminal_control_constraint(z):
            u_array, _ = self._extract_u_x(z)
            return u_array[N]  # u_N 应为 [0, 0]

        constraints.append({
            'type': 'eq',
            'fun': terminal_control_constraint
        })

        return constraints

    def solve(self, init_guess: Optional[npt.NDArray] = None) -> tuple:
        """求解最优控制问题

        Parameters
        ----------
        init_guess : Optional[npt.NDArray]
            初始猜测值

        Returns
        -------
        tuple
            (xks, uks) 其中 xks 形状为 (N+1, x_dim), uks 形状为 (N+1, u_dim)
        """
        if init_guess is None:
            init_guess = 0.01 * np.ones(self.n_vars)

        bounds = self._build_bounds()
        constraints = self._build_constraints()

        # 使用 SLSQP 方法求解
        result = minimize(
            self.J,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': 1e-9,
                'disp': True,
                'maxiter': 1000
            }
        )

        # 提取结果
        u_array, x_array = self._extract_u_x(result.x)

        return x_array, u_array
