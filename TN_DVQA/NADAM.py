from __future__ import annotations
import numpy as np
from collections.abc import Callable
from typing import Any
from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult
from quimb.tensor import MatrixProductState


class ADAM(Optimizer):
    """Adam optimizer supporting both vector (x0) and MPS (x1)."""

    def __init__(
        self,
        maxiter: int = 1000,
        tol: float = 1e-6,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
    ):
        super().__init__()
        self._maxiter = maxiter
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad

    def get_support_level(self):
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def _update_array(self, grad, m, v, v_eff, lr_eff):
        """Adam update for numpy arrays."""
        m_new = self._beta_1 * m + (1 - self._beta_1) * grad
        v_new = self._beta_2 * v + (1 - self._beta_2) * (grad * grad)

        if self._amsgrad:
            v_eff_new = np.maximum(v_eff, v_new)
        else:
            v_eff_new = v_new

        step = -lr_eff * m_new / (np.sqrt(v_eff_new) + self._noise_factor)
        return step, m_new, v_new, v_eff_new
    def minimize(
        self,
        fun: Callable[[np.ndarray, MatrixProductState], float],
        jac: Callable[[np.ndarray, MatrixProductState], tuple[np.ndarray, Any]],
        x0: np.ndarray,
        x1: MatrixProductState,
    ) -> OptimizerResult:
        """Minimize f(x0, x1). grad_mps can be list-of-arrays or MatrixProductState."""

       
        grad_vec, grad_mps = jac(x0, x1)


        if isinstance(grad_mps, MatrixProductState):
            grad_list = [T.data for T in grad_mps.tensors]
        else:
            grad_list = grad_mps


        t = 0
        m_vec = np.zeros_like(grad_vec)
        v_vec = np.zeros_like(grad_vec)
        m_mps = [np.zeros_like(g) for g in grad_list]
        v_mps = [np.zeros_like(g) for g in grad_list]

        if self._amsgrad:
            v_eff_vec = np.zeros_like(grad_vec)
            v_eff_mps = [np.zeros_like(g) for g in grad_list]
        else:
            v_eff_vec = None
            v_eff_mps = [None] * len(grad_list)


        params_vec = x0.copy()
        params_mps = x1.copy()

    
        while t < self._maxiter:
            if t > 0:
                grad_vec, grad_mps = jac(params_vec, params_mps)
                if isinstance(grad_mps, MatrixProductState):
                    grad_list = [T.data for T in grad_mps.tensors]
                else:
                    grad_list = grad_mps

    
            if t % 10 == 0:
                val = fun(params_vec, params_mps)
                print(f"Iteration {t}: Objective = {val:.6f}")

            t += 1
            lr_eff = self._lr * np.sqrt(1 - self._beta_2**t) / (1 - self._beta_1**t)


            step_vec, m_vec, v_vec, v_eff_vec = self._update_array(
                grad_vec, m_vec, v_vec, v_eff_vec, lr_eff
            )
            params_vec_new = params_vec + step_vec


            params_mps_new = params_mps.copy()
            for i, (T, g) in enumerate(zip(params_mps.tensors, grad_list)):
                step, m_mps[i], v_mps[i], v_eff_mps[i] = self._update_array(
                    g, m_mps[i], v_mps[i], v_eff_mps[i], lr_eff
                )
                updated_data = T.data + step
                params_mps_new.tensors[i].modify(data=updated_data)


            if np.linalg.norm(params_vec_new - params_vec) < self._tol:
                params_vec, params_mps = params_vec_new, params_mps_new
                break

            params_vec, params_mps = params_vec_new, params_mps_new


        result = OptimizerResult()
        result.x = (params_vec, params_mps)
        result.fun = fun(params_vec, params_mps)
        result.nfev = t
        return result