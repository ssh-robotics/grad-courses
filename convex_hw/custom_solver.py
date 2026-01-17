"""
自定义优化求解器 - 不使用 scipy.optimize
实现基于增广拉格朗日法 + 梯度下降的求解器
"""
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, Optional, List


class CustomSolver:
    """
    使用增广拉格朗日法 (Augmented Lagrangian Method) 求解带约束的优化问题

    min f(x)
    s.t. g_i(x) = 0  (等式约束)
         lb <= x <= ub (边界约束)
    """

    def __init__(
        self,
        obj_func: Callable[[npt.NDArray], float],
        eq_constraints: List[Callable[[npt.NDArray], npt.NDArray]],
        bounds: Optional[tuple] = None,
        tol: float = 1e-6,
        max_iter: int = 500,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        obj_func : Callable
            目标函数 f(x)
        eq_constraints : List[Callable]
            等式约束函数列表，每个函数返回约束残差向量
        bounds : tuple
            (lower_bounds, upper_bounds) 边界约束
        tol : float
            收敛容差
        max_iter : int
            最大迭代次数
        verbose : bool
            是否打印求解过程
        """
        self.obj_func = obj_func
        self.eq_constraints = eq_constraints
        self.bounds = bounds
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _numerical_gradient(self, func: Callable, x: npt.NDArray, eps: float = 1e-7) -> npt.NDArray:
        """数值梯度计算"""
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        return grad

    def _compute_eq_constraint_values(self, x: npt.NDArray) -> npt.NDArray:
        """计算所有等式约束的值"""
        all_constraints = []
        for cons_func in self.eq_constraints:
            c = cons_func(x)
            if np.isscalar(c):
                all_constraints.append(c)
            else:
                all_constraints.extend(c.flatten())
        return np.array(all_constraints)

    def _augmented_lagrangian(
        self,
        x: npt.NDArray,
        lambda_eq: npt.NDArray,
        mu: float
    ) -> float:
        """
        增广拉格朗日函数:
        L_A(x, λ, μ) = f(x) + λ^T * g(x) + (μ/2) * ||g(x)||^2
        """
        f_val = self.obj_func(x)
        g_val = self._compute_eq_constraint_values(x)

        # 拉格朗日项 + 二次惩罚项
        lagrangian = f_val + np.dot(lambda_eq, g_val) + (mu / 2) * np.dot(g_val, g_val)
        return lagrangian

    def _project_to_bounds(self, x: npt.NDArray) -> npt.NDArray:
        """将 x 投影到边界约束内"""
        if self.bounds is None:
            return x
        lb, ub = self.bounds
        return np.clip(x, lb, ub)

    def _line_search(
        self,
        x: npt.NDArray,
        direction: npt.NDArray,
        lambda_eq: npt.NDArray,
        mu: float,
        alpha_init: float = 1.0,
        c1: float = 1e-4,
        rho: float = 0.5
    ) -> float:
        """Armijo 回溯线搜索"""
        alpha = alpha_init
        f0 = self._augmented_lagrangian(x, lambda_eq, mu)
        grad = self._numerical_gradient(
            lambda z: self._augmented_lagrangian(z, lambda_eq, mu), x
        )
        slope = np.dot(grad, direction)

        for _ in range(20):
            x_new = self._project_to_bounds(x + alpha * direction)
            f_new = self._augmented_lagrangian(x_new, lambda_eq, mu)
            if f_new <= f0 + c1 * alpha * slope:
                break
            alpha *= rho

        return alpha

    def _inner_minimize(
        self,
        x0: npt.NDArray,
        lambda_eq: npt.NDArray,
        mu: float,
        inner_max_iter: int = 100,
        inner_tol: float = 1e-5
    ) -> npt.NDArray:
        """
        使用梯度下降最小化增广拉格朗日函数
        """
        x = x0.copy()

        for k in range(inner_max_iter):
            # 计算梯度
            grad = self._numerical_gradient(
                lambda z: self._augmented_lagrangian(z, lambda_eq, mu), x
            )

            grad_norm = np.linalg.norm(grad)
            if grad_norm < inner_tol:
                break

            # 搜索方向（负梯度方向）
            direction = -grad / (grad_norm + 1e-10)

            # 线搜索确定步长
            alpha = self._line_search(x, direction, lambda_eq, mu)

            # 更新
            x_new = self._project_to_bounds(x + alpha * direction)

            # 检查收敛
            if np.linalg.norm(x_new - x) < inner_tol:
                x = x_new
                break

            x = x_new

        return x

    def solve(self, x0: npt.NDArray) -> tuple:
        """
        求解优化问题

        Parameters
        ----------
        x0 : npt.NDArray
            初始猜测

        Returns
        -------
        tuple
            (最优解, 目标函数值, 约束违反量)
        """
        x = x0.copy()

        # 初始化拉格朗日乘子
        n_eq = len(self._compute_eq_constraint_values(x))
        lambda_eq = np.zeros(n_eq)

        # 惩罚参数
        mu = 10.0
        mu_max = 1e8
        gamma = 2.0  # 惩罚参数增长因子

        if self.verbose:
            print("Custom Solver: Augmented Lagrangian Method")
            print("=" * 60)

        prev_constraint_norm = np.inf

        for outer_iter in range(self.max_iter):
            # 内层优化：固定 λ 和 μ，最小化增广拉格朗日函数
            x = self._inner_minimize(x, lambda_eq, mu, inner_max_iter=50)

            # 计算约束违反量
            g_val = self._compute_eq_constraint_values(x)
            constraint_norm = np.linalg.norm(g_val)
            f_val = self.obj_func(x)

            if self.verbose and outer_iter % 10 == 0:
                print(f"Iter {outer_iter:3d}: f(x) = {f_val:.6f}, ||g(x)|| = {constraint_norm:.6e}, mu = {mu:.1e}")

            # 检查收敛
            if constraint_norm < self.tol:
                if self.verbose:
                    print("=" * 60)
                    print(f"Converged at iteration {outer_iter}")
                    print(f"Final objective value: {f_val:.6f}")
                    print(f"Constraint violation: {constraint_norm:.6e}")
                break

            # 更新拉格朗日乘子
            lambda_eq = lambda_eq + mu * g_val

            # 更新惩罚参数
            if constraint_norm > 0.25 * prev_constraint_norm:
                mu = min(gamma * mu, mu_max)

            prev_constraint_norm = constraint_norm

        return x, f_val, constraint_norm


class OptControlCustom:
    """使用自定义求解器的最优控制问题求解器"""

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
        self.N = N
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.J = J
        self.dyn_cons = dyn_cons
        self.x0_constraint = x0
        self.xN_constraint = xN
        self.bounds_dict = lower_upper_bound_ux
        self.n_vars = (N + 1) * (u_dim + x_dim)

    def _extract_u_x(self, z: npt.NDArray) -> tuple:
        """从优化变量向量中提取 u 和 x 序列"""
        N = self.N
        u_dim = self.u_dim
        x_dim = self.x_dim

        u_array = np.zeros((N + 1, u_dim))
        for i in range(u_dim):
            u_array[:, i] = z[i * (N + 1): (i + 1) * (N + 1)]

        x_start = u_dim * (N + 1)
        x_array = np.zeros((N + 1, x_dim))
        for i in range(x_dim):
            x_array[:, i] = z[x_start + i * (N + 1): x_start + (i + 1) * (N + 1)]

        return u_array, x_array

    def _build_bounds(self) -> tuple:
        """构建边界约束"""
        N = self.N
        u_dim = self.u_dim
        x_dim = self.x_dim

        lb_u = self.bounds_dict["lb_u"]
        ub_u = self.bounds_dict["ub_u"]
        lb_x = self.bounds_dict["lb_x"]
        ub_x = self.bounds_dict["ub_x"]

        lower_bounds = []
        upper_bounds = []

        for i in range(u_dim):
            lower_bounds.extend([lb_u[i]] * (N + 1))
            upper_bounds.extend([ub_u[i]] * (N + 1))

        for i in range(x_dim):
            lower_bounds.extend([lb_x[i]] * (N + 1))
            upper_bounds.extend([ub_x[i]] * (N + 1))

        return (np.array(lower_bounds), np.array(upper_bounds))

    def _build_eq_constraints(self) -> List[Callable]:
        """构建等式约束函数列表"""
        N = self.N

        constraints = []

        # 1. 状态转移约束
        def dynamics_constraint(z):
            u_array, x_array = self._extract_u_x(z)
            residuals = []
            for k in range(N):
                xk = x_array[k]
                xkp1 = x_array[k + 1]
                uk = u_array[k]
                ukp1 = u_array[k + 1]
                res = self.dyn_cons(xk, xkp1, uk, ukp1)        # 优化变量结构: [u_1(0:N), u_2(0:N), x_1(0:N), x_2(0:N), ..., x_5(0:N)]
        # 即先存所有时刻的 u1，再存所有时刻的 u2，再存所有时刻的 x1，以此类推
        # 总变量数: (N+1) * (u_dim + x_dim)
                residuals.extend(res)
            return np.array(residuals)

        constraints.append(dynamics_constraint)

        # 2. 初始状态约束
        def initial_constraint(z):
            _, x_array = self._extract_u_x(z)
            return x_array[0] - self.x0_constraint

        constraints.append(initial_constraint)

        # 3. 末端状态约束
        def terminal_state_constraint(z):
            _, x_array = self._extract_u_x(z)
            return x_array[N] - self.xN_constraint

        constraints.append(terminal_state_constraint)

        # 4. 末端控制约束
        def terminal_control_constraint(z):
            u_array, _ = self._extract_u_x(z)
            return u_array[N]

        constraints.append(terminal_control_constraint)

        return constraints

    def solve(self, init_guess: Optional[npt.NDArray] = None) -> tuple:
        """求解最优控制问题"""
        if init_guess is None:
            init_guess = 0.01 * np.ones(self.n_vars)

        bounds = self._build_bounds()
        eq_constraints = self._build_eq_constraints()

        solver = CustomSolver(
            obj_func=self.J,
            eq_constraints=eq_constraints,
            bounds=bounds,
            tol=1e-4,
            max_iter=200,
            verbose=True
        )

        x_opt, f_val, constraint_viol = solver.solve(init_guess)

        u_array, x_array = self._extract_u_x(x_opt)

        return x_array, u_array
