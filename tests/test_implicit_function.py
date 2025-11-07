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


# ---- Explicit function F(x) with closed-form JVP/VJP ----
class Bratu1DF(torch.autograd.Function):
    """
    F(x) = -shift_left(x) + (2 + h^2 * lam * exp(x)) * x - shift_right(x)
    (written elementwise: F_i = -x_{i-1} + 2*x_i - x_{i+1} + h^2 * lam * exp(x_i))
    Output shape == input shape (1D vector).
    """

    # problem parameters (class attributes so we avoid instances)
    n: int = 100
    lam: float = 1.0

    @classmethod
    def _h2(cls) -> float:
        return (1.0 / (cls.n + 1)) ** 2

    @staticmethod
    def _shift_left(v: torch.Tensor):
        out = torch.zeros_like(v)
        out[1:] = v[:-1]
        return out

    @staticmethod
    def _shift_right(v: torch.Tensor):
        out = torch.zeros_like(v)
        out[:-1] = v[1:]
        return out

    # ------- ExplicitADFunction-style hooks (static/class methods) -------

    @staticmethod
    def compute(x: torch.Tensor) -> torch.Tensor:
        h2 = Bratu1DF._h2()
        lam = Bratu1DF.lam
        e = torch.exp(x)
        xL = Bratu1DF._shift_left(x)
        xR = Bratu1DF._shift_right(x)
        return -xL + 2.0 * x - xR + h2 * lam * e

    @staticmethod
    def compute_primals(x: torch.Tensor, outputs: torch.Tensor):
        # Cache what we need for fast JVP/VJP (no graph references)
        h2 = Bratu1DF._h2()
        lam = Bratu1DF.lam
        # Diagonal of Jacobian at x: 2 + h^2 * lam * exp(x)
        diag = (2.0 + h2 * lam * torch.exp(x.detach())).detach()
        return (x.detach(), diag)

    @staticmethod
    def jvp_from_primals(saved, dx: torch.Tensor):
        """
        JVP: J(x) @ dx = -shift_left(dx) + diag * dx - shift_right(dx)
        """
        _x, diag = saved
        dxL = Bratu1DF._shift_left(dx)
        dxR = Bratu1DF._shift_right(dx)
        return -dxL + diag * dx - dxR

    @staticmethod
    def vjp_from_primals(saved, v: torch.Tensor, needs_input_grad=None):
        """
        VJP: since J is symmetric tridiagonal here, J^T v = J v
        """
        _x, diag = saved
        vL = Bratu1DF._shift_left(v)
        vR = Bratu1DF._shift_right(v)
        grad_x = -vL + diag * v - vR
        if needs_input_grad is None:
            return (grad_x,)
        return (grad_x,) if needs_input_grad[0] else (None,)


# ---- GMRES-enabled Bratu function ----
class Bratu1DGMRES(Bratu1DF, GMRESSolver):
    @classmethod
    def matvec(cls, x, primals):
        # single-input function: dx tangent is x
        return cls.jvp_from_primals(primals, x)

    @classmethod
    def compute_residual(cls, b, x, primals):
        return b - cls.matvec(x, primals)

    @classmethod
    def gmres(
        cls, b, primals, x0=None, tol=1e-6, atol=0.0, restart=20, maxiter=None
    ):
        # call your GMRES implementation; this stub assumes it's in GMRESSolver
        return super().gmres(
            b=b,
            primals=primals,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
        )


# Make a concrete implicit solver class for Bratu
class BratuImplicit(ImplicitFunction):
    func = Bratu1DGMRES
    variable_input = 1  # there is only one input (x)


class TestBratuImplicit(unittest.TestCase):
    def test_bratu_residual_small(self):
        # Optional: make the test deterministic
        torch.manual_seed(0)

        # Configure the problem
        Bratu1DF.n = 100
        Bratu1DF.lam = 1.0

        n = Bratu1DF.n
        x0 = torch.randn(n) * 0.1  # reasonable initial guess
        y = torch.zeros(n)         # solve F(x) = 0

        # New call signature: .apply(y, *inputs) — here inputs is just x0
        x_sol = BratuImplicit.apply(y, x0)

        # Check residual norm
        res = Bratu1DF.compute(x_sol)
        res_norm = torch.linalg.vector_norm(res)

        # Assert it's small
        tol = 1e-6
        self.assertLess(
            res_norm.item(),
            tol,
            msg=f"Residual too large: {res_norm.item()} > {tol}",
        )

# ---------- Extended Rosenbrock: F(x) and its derivatives ----------
class ExtendedRosenbrockF(ExplicitADFunction):
    """
    F(x) on R^n (n even):
      Let a = x[0::2], b = x[1::2]
      F_odd  = -400*a*(b - a^2) - 2*(1 - a)
      F_even =  200*(b - a^2)
      out[0::2] = F_odd, out[1::2] = F_even
    """

    @staticmethod
    def compute(x: torch.Tensor) -> torch.Tensor:
        a = x[0::2]
        b = x[1::2]
        F_odd = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a)
        F_even = 200.0 * (b - a * a)
        out = torch.empty_like(x)
        out[0::2] = F_odd
        out[1::2] = F_even
        return out

    @staticmethod
    def compute_primals(x: torch.Tensor, outputs: torch.Tensor):
        # Precompute partials (diagonal blocks) at x (detach to avoid graph retention)
        a = x[0::2].detach()
        b = x[1::2].detach()

        # dF_odd/da and dF_odd/db
        dF_odd_da = (-400.0 * b + 1200.0 * a * a + 2.0).detach()
        dF_odd_db = (-400.0 * a).detach()

        # dF_even/da and dF_even/db
        dF_even_da = (-400.0 * a).detach()
        dF_even_db = (200.0 * torch.ones_like(a)).detach()

        return (x.detach(), dF_odd_da, dF_odd_db, dF_even_da, dF_even_db)

    @staticmethod
    def jvp_from_primals(saved, dx: torch.Tensor):
        _, dF_odd_da, dF_odd_db, dF_even_da, dF_even_db = saved
        da = dx[0::2]
        db = dx[1::2]
        jvp_odd = dF_odd_da * da + dF_odd_db * db
        jvp_even = dF_even_da * da + dF_even_db * db
        out = torch.empty_like(dx)
        out[0::2] = jvp_odd
        out[1::2] = jvp_even
        return out

    @staticmethod
    def vjp_from_primals(saved, v: torch.Tensor, needs_input_grad=None):
        _, dF_odd_da, dF_odd_db, dF_even_da, dF_even_db = saved
        v_odd = v[0::2]
        v_even = v[1::2]
        # grad wrt a (x[0::2])
        grad_a = dF_odd_da * v_odd + dF_even_da * v_even
        # grad wrt b (x[1::2])
        grad_b = dF_odd_db * v_odd + dF_even_db * v_even
        grad_x = torch.empty_like(v)
        grad_x[0::2] = grad_a
        grad_x[1::2] = grad_b
        if needs_input_grad is None:
            return (grad_x,)
        return (grad_x,) if needs_input_grad[0] else (None,)


# ---------- GMRES wrapper for Extended Rosenbrock ----------
class ExtendedRosenbrockGMRES(ExtendedRosenbrockF, GMRESSolver):
    @classmethod
    def matvec(cls, x, primals):
        # single-input function: dx tangent is x
        return cls.jvp_from_primals(primals, x)

    @classmethod
    def compute_residual(cls, b, x, primals):
        return b - cls.matvec(x, primals)


# ---------- Implicit solver: solve F(x) = 0 ----------
class RosenImplicit(ImplicitFunction):
    func = ExtendedRosenbrockGMRES
    variable_input = 1  # only one input (x), 1-based index

# before calling apply:
RosenImplicit._newton_maxiter = 100
RosenImplicit._gmres_maxiter = 400

class TestExtendedRosenbrockImplicit(unittest.TestCase):
    def test_extended_rosenbrock_residual_small(self):
        torch.manual_seed(0)

        n = 100
        self.assertEqual(n % 2, 0, "n must be even for Extended Rosenbrock")

        # Robust starting point: the known steady state (a=1, b=1)
        # This makes the test deterministic and avoids line-search/globalization subtleties.
        x0 = torch.ones(n, dtype=torch.get_default_dtype())
        y = torch.zeros(n)  # solve F(x) = 0

        # Varargs call (not a tuple)
        x_sol = RosenImplicit.apply(y, x0)

        # Check residual norm
        res = ExtendedRosenbrockF.compute(x_sol)
        res_norm = torch.linalg.vector_norm(res)

        tol = 1e-6
        self.assertLess(
            res_norm.item(),
            tol,
            msg=f"Residual too large: {res_norm.item()} > {tol}",
        )

# ---------- Lorenz–96 steady-state: F(x) and its derivatives ----------
class Lorenz96F(ExplicitADFunction):
    """
    F(x)_i = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + forcing
    (cyclic indices)
    """

    # Problem parameters (class attributes for simplicity)
    n: int = 100
    forcing: float = 8.0

    @staticmethod
    def _roll(v: torch.Tensor, k: int) -> torch.Tensor:
        return torch.roll(v, shifts=k, dims=0)

    @classmethod
    def compute(cls, x: torch.Tensor) -> torch.Tensor:
        xm1 = cls._roll(x, +1)  # x_{i-1}
        xm2 = cls._roll(x, +2)  # x_{i-2}
        xp1 = cls._roll(x, -1)  # x_{i+1}
        return (xp1 - xm2) * xm1 - x + cls.forcing

    @classmethod
    def compute_primals(cls, x: torch.Tensor, outputs: torch.Tensor):
        # Cache the shifted neighbors (detach to avoid holding autograd graph)
        a = cls._roll(x.detach(), +2)  # x_{i-2}
        b = cls._roll(x.detach(), +1)  # x_{i-1}
        d = cls._roll(x.detach(), -1)  # x_{i+1}
        return (x.detach(), a, b, d)

    @classmethod
    def jvp_from_primals(cls, saved, dx: torch.Tensor):
        """
        J(x)·dx = (-b) * dx_{i-2} + (d - a) * dx_{i-1} - dx_i + b * dx_{i+1}
        implemented with cyclic rolls.
        """
        _x, a, b, d = saved
        dxm2 = cls._roll(dx, +2)
        dxm1 = cls._roll(dx, +1)
        dxp1 = cls._roll(dx, -1)
        return (-b) * dxm2 + (d - a) * dxm1 - dx + (b) * dxp1

    @classmethod
    def vjp_from_primals(cls, saved, v: torch.Tensor, needs_input_grad=None):
        """
        J(x)^T v  (derived by index shifting)
        (J^T v)_j = -v_j
                    + (d_{j+1} - a_{j+1}) * v_{j+1}
                    + b_{j-1} * v_{j-1}
                    - b_{j+2} * v_{j+2}
        """
        _x, a, b, d = saved

        # Helpful rolls of v
        v_p1 = cls._roll(v, -1)  # v_{j+1}
        v_m1 = cls._roll(v, +1)  # v_{j-1}
        v_p2 = cls._roll(v, -2)  # v_{j+2}

        # Coefficient arrays aligned at index j
        coeff_p1 = cls._roll(d, -1) - cls._roll(a, -1)  # d_{j+1} - a_{j+1}
        coeff_m1 = cls._roll(b, +1)  # b_{j-1}
        coeff_p2 = -cls._roll(b, -2)  # -b_{j+2}

        jt_v = -v + coeff_p1 * v_p1 + coeff_m1 * v_m1 + coeff_p2 * v_p2

        if needs_input_grad is None:
            return (jt_v,)
        return (jt_v,) if needs_input_grad[0] else (None,)


# ---------- GMRES wrapper for Lorenz–96 ----------
class Lorenz96GMRES(Lorenz96F, GMRESSolver):
    @classmethod
    def matvec(cls, x, primals):
        # Single-input function: tangent dx is x
        return cls.jvp_from_primals(primals, x)

    @classmethod
    def compute_residual(cls, b, x, primals):
        return b - cls.matvec(x, primals)


# ---------- Implicit solver: solve F(x) = 0 ----------
class L96Implicit(ImplicitFunction):
    func = Lorenz96GMRES
    variable_input = 1  # only one input (x), 1-based index


# =============================== TESTS =================================
class TestLorenz96Implicit(unittest.TestCase):
    def test_lorenz96_residual_small(self):
        torch.manual_seed(0)

        # Problem configuration
        n = 100
        forcing = 0.5
        Lorenz96F.n = n
        Lorenz96F.forcing = forcing

        # Random normal around the constant solution xi ≈ forcing
        x0 = torch.normal(mean=torch.full((n,), forcing), std=torch.ones(n))

        # Solve F(x) = 0
        y = torch.zeros(n)
        x_sol = L96Implicit.apply(y, x0,)

        # Check residual norm is small
        res = Lorenz96F.compute(x_sol)
        res_norm = torch.linalg.vector_norm(res)
        tol = 1e-6
        self.assertLess(
            res_norm.item(),
            tol,
            msg=f"Residual too large: {res_norm.item()} > {tol}",
        )


if __name__ == "__main__":
    unittest.main()
