import importlib
import unittest
import torch


_core = importlib.import_module("heteromarket.core")
find_equilibrium_prices = _core.find_equilibrium_prices


def _as_float(x):
    # Accepts Tensor or Python number
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)

class TestDirectionalGradientsExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)

    def setUp(self):
        # Problem with broken symmetry (deterministic)
        self.B, self.N = 2, 2

        # Fixed inputs (require grads for the targets we differentiate)
        self.Sigma = torch.tensor(
            [
                [[0.7086, 0.2066], [0.2066, 0.5898]],
                [[0.4318, 0.6247], [0.6247, 2.6645]],
            ],
            requires_grad=True,
            dtype=torch.float64,
        )
        self.mu = torch.tensor(
            [[2.0486, 2.3718], [2.0554, 2.3971]],
            requires_grad=True,
            dtype=torch.float64,
        )
        self.com = torch.zeros(self.B, dtype=torch.float64).requires_grad_(
            True
        )
        self.X = torch.zeros(
            (self.B, self.N), dtype=torch.float64
        ).requires_grad_(True)
        self.budget = torch.tensor(
            [1.0, 1.3], dtype=torch.float64, requires_grad=True
        )
        self.kappa = torch.tensor(
            [0.15, 0.05], dtype=torch.float64, requires_grad=True
        )  # short caps
        self.theta = torch.tensor(
            [1.2, 1.6], dtype=torch.float64, requires_grad=True
        )  # long caps
        self.S = torch.linspace(
            0.7, 1.3, self.N, dtype=torch.float64
        ).requires_grad_(True)

        # Starting guess for prices
        self.p0 = torch.ones(self.N, dtype=torch.float64)

        # Fixed probing vector for the scalar functional <p_eq, v>
        self.v = torch.tensor([0.3428, -1.0417], dtype=torch.float64)

        # Small FD step
        self.eps = 3e-8

        # Relative and absolute tolerances
        self.rel_tol = 3e-4
        self.abs_tol = 5e-7  # used when true derivative magnitude is tiny

    # ---------- helpers ----------
    def _eq_prices(self, Sigma, mu, com, X, budget, kappa, theta, S, p0):
        return find_equilibrium_prices(
            Sigma,
            mu,
            com,
            X,
            budget,
            kappa,
            theta,
            S,
            initial_approximation=p0,
        )

    def _scalar_loss(self, p):
        # Loss = <p, v>
        return (p * self.v).sum()

    @staticmethod
    def _fd_dir(fn, x, dx, eps):
        return (fn(x + eps * dx) - fn(x - eps * dx)) / (2.0 * eps)

    # ---------- tests ----------
    def test_budget_directional_derivative(self):
        # Forward
        p_eq = self._eq_prices(
            self.Sigma,
            self.mu,
            self.com,
            self.X,
            self.budget,
            self.kappa,
            self.theta,
            self.S,
            self.p0,
        )
        loss = self._scalar_loss(p_eq)

        # AD gradient wrt budget
        (g_budget,) = torch.autograd.grad(
            loss, (self.budget,), allow_unused=True
        )

        # FD directional derivative wrt budget
        Db = torch.tensor([-0.3, 0.2], dtype=torch.float64)

        def f_budget(bud):
            p = self._eq_prices(
                self.Sigma,
                self.mu,
                self.com,
                self.X,
                bud,
                self.kappa,
                self.theta,
                self.S,
                self.p0,
            )
            return self._scalar_loss(p)

        fd_dir = self._fd_dir(f_budget, self.budget, Db, self.eps)
        ad_dir = (g_budget * Db).sum()

        # Compare with relative error; fall back to abs if FD ~ 0
        if fd_dir.abs() > 1e-8:
            rel = (ad_dir - fd_dir).abs() / fd_dir.abs()
            self.assertLess(_as_float(rel), self.rel_tol)
        else:
            self.assertLess(_as_float((ad_dir - fd_dir).abs()), self.abs_tol)

    def test_kappa_directional_derivative(self):
        p_eq = self._eq_prices(
            self.Sigma,
            self.mu,
            self.com,
            self.X,
            self.budget,
            self.kappa,
            self.theta,
            self.S,
            self.p0,
        )
        loss = self._scalar_loss(p_eq)

        (g_kappa,) = torch.autograd.grad(
            loss, (self.kappa,), allow_unused=True
        )

        Dk = torch.tensor([0.25, -0.15], dtype=torch.float64)

        def f_kappa(ka):
            p = self._eq_prices(
                self.Sigma,
                self.mu,
                self.com,
                self.X,
                self.budget,
                ka,
                self.theta,
                self.S,
                self.p0,
            )
            return self._scalar_loss(p)

        fd_dir = self._fd_dir(f_kappa, self.kappa, Dk, self.eps)
        ad_dir = (g_kappa * Dk).sum()

        if fd_dir.abs() > 1e-8:
            rel = (ad_dir - fd_dir).abs() / fd_dir.abs()
            self.assertLess(_as_float(rel), self.rel_tol)
        else:
            self.assertLess(_as_float((ad_dir - fd_dir).abs()), self.abs_tol)

    def test_mu_directional_derivative(self):
        p_eq = self._eq_prices(
            self.Sigma,
            self.mu,
            self.com,
            self.X,
            self.budget,
            self.kappa,
            self.theta,
            self.S,
            self.p0,
        )
        loss = self._scalar_loss(p_eq)

        (g_mu,) = torch.autograd.grad(loss, (self.mu,), allow_unused=True)

        g = torch.Generator().manual_seed(123)
        Dmu = torch.randn(self.mu.shape, generator=g, dtype=torch.float64)

        def f_mu(mu_in):
            p = self._eq_prices(
                self.Sigma,
                mu_in,
                self.com,
                self.X,
                self.budget,
                self.kappa,
                self.theta,
                self.S,
                self.p0,
            )
            return self._scalar_loss(p)

        fd_dir = self._fd_dir(f_mu, self.mu, Dmu, self.eps)
        ad_dir = (g_mu * Dmu).sum()

        if fd_dir.abs() > 1e-8:
            rel = (ad_dir - fd_dir).abs() / fd_dir.abs()
            self.assertLess(_as_float(rel), self.rel_tol)
        else:
            self.assertLess(_as_float((ad_dir - fd_dir).abs()), self.abs_tol)

    def test_sigma_directional_derivative_pd_preserving(self):
        p_eq = self._eq_prices(
            self.Sigma,
            self.mu,
            self.com,
            self.X,
            self.budget,
            self.kappa,
            self.theta,
            self.S,
            self.p0,
        )
        loss = self._scalar_loss(p_eq)

        (g_Sigma,) = torch.autograd.grad(
            loss, (self.Sigma,), allow_unused=True
        )

        # Symmetric direction; use torch.randn(shape, generator=...)
        gS = torch.Generator().manual_seed(456)
        R = torch.randn(self.Sigma.shape, generator=gS, dtype=torch.float64)
        DS = 0.5 * (R + R.transpose(-1, -2))  # symmetric

        # Keep QP well-posed: add tiny ridge to both ± evaluations
        ridge = 1e-6 * torch.eye(self.N, dtype=torch.float64).expand(
            self.B, self.N, self.N
        )

        def f_Sigma(Sig_in):
            p = self._eq_prices(
                Sig_in + ridge,
                self.mu,
                self.com,
                self.X,
                self.budget,
                self.kappa,
                self.theta,
                self.S,
                self.p0,
            )
            return self._scalar_loss(p)

        fd_dir = self._fd_dir(f_Sigma, self.Sigma, DS, self.eps)
        ad_dir = (g_Sigma * DS).sum()

        if fd_dir.abs() > 1e-8:
            rel = (ad_dir - fd_dir).abs() / fd_dir.abs()
            self.assertLess(_as_float(rel), self.rel_tol)
        else:
            self.assertLess(_as_float((ad_dir - fd_dir).abs()), self.abs_tol)


if __name__ == "__main__":
    unittest.main()
