import importlib
import unittest
import torch

_core = importlib.import_module("heteromarket.core")
ExplicitADFunction = _core.ExplicitADFunction
GMRESSolver = _core.GMRESSolver
ImplicitFunction = _core.ImplicitFunction

# -------- Concrete F and its GMRES wrapper used in tests --------
class MyFunction(ExplicitADFunction):
    @staticmethod
    def compute(x1, x2):
        # Elementwise: y = x1^2 + 3*x2
        return x1**2 + 3 * x2

    @staticmethod
    def compute_primals(x1, x2, outputs):
        # cache inputs (detached) – sufficient for closed-form rules
        return (x1.detach(), x2.detach())

    @staticmethod
    def vjp_from_primals(saved, bar_y, needs_input_grad=None):
        x1, x2 = saved
        gx1 = 2 * x1 * bar_y if bar_y is not None else None
        gx2 = 3 * bar_y     if bar_y is not None else None
        grads = (gx1, gx2)
        if needs_input_grad is None:
            return grads
        return tuple(g if need else None for g, need in zip(grads, needs_input_grad))

    @staticmethod
    def jvp_from_primals(saved, dx1, dx2):
        x1, x2 = saved
        dx1 = dx1 if dx1 is not None else 0
        dx2 = dx2 if dx2 is not None else 0
        return 2 * x1 * dx1 + 3 * dx2


class MyFunctionGMRES(MyFunction, GMRESSolver):
    @classmethod
    def matvec(cls, x, primals):
        # compute J_x1 @ x via JVP: dx1=x, dx2=None
        return cls.jvp_from_primals(primals, x, None)

    @classmethod
    def compute_residual(cls, b, x, primals):
        return b - cls.matvec(x, primals)


# -------- Implicit class used in tests: solve for x1 (variable_input=1) --------
class G(ImplicitFunction):
    func = MyFunctionGMRES
    variable_input = 1  # solve for x1 in inputs = (x1, x2)

class GBad(ImplicitFunction):
    func = MyFunctionGMRES
    variable_input = 3  # but we'll pass only 2 inputs

# =============================== TESTS =================================
class TestImplicitFunction(unittest.TestCase):

    def test_forward_solve_scalar(self):
        # y = x1^2 + 3*x2  ->  x1 = sqrt(y - 3*x2)
        y = torch.tensor([10.0])
        x1_init = torch.tensor([1.0])      # initial guess
        x2 = torch.tensor([2.0])

        # New call signature: .apply(y, *inputs)
        x1_sol = G.apply(y, x1_init, x2)

        expected = torch.sqrt(y - 3 * x2)
        self.assertTrue(torch.allclose(x1_sol, expected, atol=1e-6, rtol=0),
                        msg=f"x1_sol={x1_sol}, expected={expected}")

    def test_compute_primals_caches_solution(self):
        y = torch.tensor([10.0])
        x1_init = torch.tensor([1.0])
        x2 = torch.tensor([2.0])

        x1_sol = G.apply(y, x1_init, x2)

        # New signature: compute_primals(y, *inputs, outputs=...)
        saved = G.compute_primals(y, x1_init, x2, outputs=x1_sol)

        self.assertIn("primals_F", saved)
        self.assertIn("inputs_sol", saved)
        self.assertEqual(saved["var_i"], 0)
        # inputs_sol contains solved x1 at position 0
        self.assertTrue(torch.allclose(saved["inputs_sol"][0], x1_sol))

        # y consistency
        y_check = G.func.compute(*saved["inputs_sol"])
        self.assertTrue(torch.allclose(y_check, y, atol=1e-6))

    def test_vjp_implicit(self):
        # Analytic check:
        # J = dF/dx1 = 2*x1 ; J^T w = bar_x1 -> w = bar_x1 / (2*x1)
        # bar_y = w ; bar_x2 = -(dF/dx2)^T w = -3*w
        y = torch.tensor([10.0])
        x1_init = torch.tensor([1.0])
        x2 = torch.tensor([2.0])

        # Solve and cache
        x1_sol = G.apply(y, x1_init, x2)
        saved = G.compute_primals(y, x1_init, x2, outputs=x1_sol)

        bar_xvar = torch.ones_like(x1_sol)  # upstream cotangent on x1 (the output)
        # New return: (bar_y, *bar_inputs)
        bar_y, bar_x1_in, bar_x2_in = G.vjp_from_primals(saved, bar_xvar)

        w_expected = bar_xvar / (2 * x1_sol)
        self.assertTrue(torch.allclose(bar_y, w_expected, atol=1e-6))

        # Grad wrt the variable input argument is None (initial guess, not a true input)
        self.assertIsNone(bar_x1_in)
        # Grad wrt x2 input should be -3*w
        self.assertTrue(torch.allclose(bar_x2_in, -3 * w_expected, atol=1e-6))

    def test_bad_variable_index_raises(self):
        y = torch.tensor([1.0])
        with self.assertRaises(IndexError):
            # New call signature: .apply(y, *inputs)
            GBad.apply(y, torch.tensor([0.0]), torch.tensor([0.0]))

    def test_shape_mismatch_raises(self):
        y = torch.tensor([10.0, 11.0])  # shape (2,)
        x1_init = torch.tensor([0.0])   # shape (1,)
        x2 = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            # New call signature: .apply(y, *inputs)
            G.apply(y, x1_init, x2)

if __name__ == "__main__":
    unittest.main()
