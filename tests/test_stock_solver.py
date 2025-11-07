import importlib
import unittest
import torch
import numpy as np
from scipy import stats
from torch.autograd.functional import jvp
import itertools

_core = importlib.import_module("heteromarket.core")
StockSolverFunc = _core.StockSolverFunc
bdot = _core.bdot

class TestActiveSetQPSolverInit(unittest.TestCase):
    @staticmethod
    def make_data(B=3, n=4, device="cpu", dtype=torch.float64):
        Q = torch.randn(B, n, n, device=device, dtype=dtype)
        Q = 0.5 * (
            Q + Q.transpose(-2, -1)
        )  # symmetrize (not required but realistic)
        m = torch.randn(B, n, device=device, dtype=dtype)
        X = torch.full((B, n), 2.0, device=device, dtype=dtype)
        budget = torch.full((B,), 10.0, device=device, dtype=dtype)
        kappa = torch.full((B,), 0.5, device=device, dtype=dtype)
        theta = torch.full((B,), 1.5, device=device, dtype=dtype)
        wl = torch.full((B,), 0.5, device=device, dtype=dtype)
        wh = torch.full((B,), 1.5, device=device, dtype=dtype)
        c = torch.full((B,), 0.1, device=device, dtype=dtype)
        p = torch.full((n,), 1.0, device=device, dtype=dtype)
        x0 = None
        return Q, m, c, X, budget, kappa, theta, p, x0

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_device_mismatch_raises(self):
        Q, m, c, X, budget, kappa, theta, p, x0 = self.make_data(
            B=2, n=3, device="cpu"
        )
        Q_cuda = Q.to("cuda")
        with self.assertRaises(TypeError):
            StockSolverFunc.apply(Q_cuda, m, c, X, budget, kappa, theta, p, x0)

    def test_Q_shape_errors(self):
        Q, m, c, X, budget, kappa, theta, p, x0 = self.make_data(B=2, n=3)
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q[:, :2, :3], m, c, X, budget, kappa, theta, p, x0
            )  # not square
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q[0], m, c, X, budget, kappa, theta, p, x0
            )  # not 3D

    def test_vector_shape_errors(self):
        Q, m, c, X, budget, kappa, theta, p, x0 = self.make_data(B=2, n=3)
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m[:1], c, X, budget, kappa, theta, p, x0
            )  # m wrong B
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m[:, :2], c, X, budget, kappa, theta, p, x0
            )  # m wrong n
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m, c[:1], X, budget, kappa, theta, p, x0
            )  # c wrong B
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m, c, X, budget[:1], kappa, theta, p, x0
            )  # wl wrong B
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m, c, X, budget, kappa[:1], theta, p, x0
            )  # wh wrong B
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m, c, X, budget, kappa, theta[:1], p, x0
            )  # U wrong n

    def test_kappa_theta_sign_violation(self):
        Q, m, c, X, budget, kappa, theta, p, x0 = self.make_data(B=2, n=3)
        kappa_bad = kappa.clone()
        kappa_bad[0] = -3.0
        theta_bad = theta.clone()
        theta_bad[0] = -2.0  # U < L at one coord
        with self.assertRaises(ValueError):
            StockSolverFunc.apply(
                Q, m, c, X, budget, kappa_bad, theta_bad, p, x0
            )


class TestStockSolverFunc(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        # Use double precision across the board for stable finite differences.
        self.dtype = torch.float64
        self.device = torch.device("cpu")

        self.B = 2
        self.n = 3
        B, n = self.B, self.n
        D, dev = self.dtype, self.device

        # SPD Q, well-conditioned
        I = torch.eye(n, dtype=D, device=dev)
        self.Q = (0.1 * I).unsqueeze(0).repeat(B, 1, 1)

        # Mild linear term
        self.m = torch.tensor(
            [[0.02, -0.01, 0.015], [-0.01, 0.01, -0.02]], dtype=D, device=dev
        )

        # c (present but smooth-path gradient ~0)
        self.c = torch.full((B,), 0.1, dtype=D, device=dev)

        # Holdings X small
        self.X = torch.tensor(
            [[0.05, 0.02, 0.03], [0.01, 0.04, 0.02]], dtype=D, device=dev
        )

        # VERY wide feasible region so we stay strictly interior for small deltas
        self.budget = torch.full((B,), 10.0, dtype=D, device=dev)
        self.kappa = torch.full((B,), 5.0, dtype=D, device=dev)
        self.theta = torch.full((B,), 5.0, dtype=D, device=dev)

        # Prices
        self.p = torch.tensor([1.2, 1.5, 1.8], dtype=D, device=dev)

        # Start point
        self.x0 = torch.zeros((B, n), dtype=D, device=dev)

        # Finite-difference config (double precision)
        self.eps_fd = 2e-7  # small, but not too small in float64
        self.atol = 2e-6
        self.rtol = 2e-6

        # Ridge used for Q directional FD to keep SPD
        self.q_ridge = 1e-12

    # ---------- helpers ----------

    def _call(self, **kwargs):
        return StockSolverFunc.apply(
            kwargs["Q"],
            kwargs["m"],
            kwargs["c"],
            kwargs["X"],
            kwargs["budget"],
            kwargs["kappa"],
            kwargs["theta"],
            kwargs["p"],
            kwargs["x0"],
        )

    def _loss(self, x):
        # scalar objective for gradient checks
        return x.sum()

    def _args_base(self):
        return dict(
            Q=self.Q.detach(),
            m=self.m.detach(),
            c=self.c.detach(),
            X=self.X.detach(),
            budget=self.budget.detach(),
            kappa=self.kappa.detach(),
            theta=self.theta.detach(),
            p=self.p.detach(),
            x0=self.x0.detach(),
        )

    # Generic central FD with adaptive backoff & NaN-masking
    def _fd_grad_generic(self, tensor, name, eps=None, max_backoffs=10):
        if eps is None:
            eps = self.eps_fd
        num = torch.full_like(tensor, float("nan"))

        for idx in itertools.product(*[range(s) for s in tensor.shape]):
            e = eps
            val = None
            for _ in range(max_backoffs):
                t_plus = tensor.detach().clone()
                t_plus[idx] += e
                t_minus = tensor.detach().clone()
                t_minus[idx] -= e
                plus = self._args_base()
                minus = self._args_base()
                plus[name] = t_plus
                minus[name] = t_minus
                xp = self._call(**plus)
                xm = self._call(**minus)
                if torch.isfinite(xp).all() and torch.isfinite(xm).all():
                    val = (self._loss(xp) - self._loss(xm)) / (2 * e)
                    break
                e *= 0.5
            if val is not None:
                num[idx] = val
        return num

    def _assert_allclose_masked(self, a, b, msg):
        mask = torch.isfinite(a) & torch.isfinite(b)
        self.assertTrue(mask.any(), f"{msg}: no finite entries to compare")
        diff = (a - b)[mask]
        self.assertTrue(
            torch.allclose(a[mask], b[mask], atol=self.atol, rtol=self.rtol),
            msg + f" (max abs diff={diff.abs().max().item():.4e})",
        )

    # Directional FD for Q (symmetric direction), with backoff and SPD ridge
    def _directional_fd_Q(self, Q_base, D, eps=None, max_backoffs=10):
        if eps is None:
            eps = self.eps_fd
        I = torch.eye(
            Q_base.size(-1), dtype=Q_base.dtype, device=Q_base.device
        ).unsqueeze(0)
        base = self._args_base()
        base["Q"] = Q_base

        e = eps
        for _ in range(max_backoffs):
            Qp = (Q_base + e * D) + self.q_ridge * I
            Qm = (Q_base - e * D) + self.q_ridge * I
            xp = self._call(**{**base, "Q": Qp})
            xm = self._call(**{**base, "Q": Qm})
            if torch.isfinite(xp).all() and torch.isfinite(xm).all():
                return (self._loss(xp) - self._loss(xm)) / (2 * e)
            e *= 0.5
        return None

    def _random_symmetric_direction(self, B, n, scale=1e-4):
        M = torch.randn(B, n, n, dtype=self.dtype, device=self.device) * scale
        return 0.5 * (M + M.transpose(-1, -2))

    # ---------- tests ----------

    def test_forward_shapes_and_finiteness(self):
        args = self._args_base()
        x = self._call(**args)
        self.assertEqual(tuple(x.shape), (self.B, self.n))
        self.assertTrue(torch.isfinite(x).all())

    def test_backward_matches_finite_diff_all_inputs_except_Q(self):
        # Attach grads
        Q = self.Q.clone().requires_grad_(True)
        m = self.m.clone().requires_grad_(True)
        c = self.c.clone().requires_grad_(True)
        X = self.X.clone().requires_grad_(True)
        budget = self.budget.clone().requires_grad_(True)
        kappa = self.kappa.clone().requires_grad_(True)
        theta = self.theta.clone().requires_grad_(True)
        p = self.p.clone().requires_grad_(True)
        x0 = self.x0.clone().requires_grad_(True)

        x = StockSolverFunc.apply(Q, m, c, X, budget, kappa, theta, p, x0)
        self._loss(x).backward()

        # autograd grads
        gm, gc, gX = m.grad.detach(), c.grad.detach(), X.grad.detach()
        gb, gk, gt, gp, gx0 = (
            budget.grad.detach(),
            kappa.grad.detach(),
            theta.grad.detach(),
            p.grad.detach(),
            x0.grad.detach(),
        )

        # FD grads (adaptive) — EXCLUDING Q (Q checked directionally)
        fdm = self._fd_grad_generic(m.detach(), "m")
        fdc = self._fd_grad_generic(c.detach(), "c")
        fdX = self._fd_grad_generic(X.detach(), "X")
        fdb = self._fd_grad_generic(budget.detach(), "budget")
        fdk = self._fd_grad_generic(kappa.detach(), "kappa")
        fdt = self._fd_grad_generic(theta.detach(), "theta")
        fdp = self._fd_grad_generic(p.detach(), "p")
        fdx0 = self._fd_grad_generic(x0.detach(), "x0")

        self._assert_allclose_masked(gm, fdm, "m gradient mismatch")
        self._assert_allclose_masked(gc, fdc, "c gradient mismatch")
        self._assert_allclose_masked(gX, fdX, "X gradient mismatch")
        self._assert_allclose_masked(gb, fdb, "budget gradient mismatch")
        self._assert_allclose_masked(gk, fdk, "kappa gradient mismatch")
        self._assert_allclose_masked(gt, fdt, "theta gradient mismatch")
        self._assert_allclose_masked(gp, fdp, "p gradient mismatch")
        self._assert_allclose_masked(gx0, fdx0, "x0 gradient mismatch")

    def test_Q_gradient_directional_checks(self):
        """
        For Q, use directional checks in float64:
        <gQ, D> ≈ d/dε loss(Q + εD) |_{ε=0}
        """
        Q = self.Q.clone().requires_grad_(True)
        m = self.m.clone()
        c = self.c.clone()
        X = self.X.clone()
        budget = self.budget.clone()
        kappa = self.kappa.clone()
        theta = self.theta.clone()
        p = self.p.clone()
        x0 = self.x0.clone()

        x = StockSolverFunc.apply(Q, m, c, X, budget, kappa, theta, p, x0)
        self._loss(x).backward()
        gQ = Q.grad.detach()

        num_ok = 0
        trials = 6
        for _ in range(trials):
            D = self._random_symmetric_direction(self.B, self.n, scale=1e-5)
            fd = self._directional_fd_Q(Q.detach(), D)
            if fd is None:
                continue
            inner = (gQ * D).sum()
            self.assertTrue(
                torch.allclose(inner, fd, atol=self.atol, rtol=self.rtol),
                f"Q directional grad mismatch (|inner-fd|={abs((inner-fd).item()):.3e})",
            )
            num_ok += 1
        if num_ok == 0:
            self.skipTest(
                "All Q directional finite diffs failed to be finite after backoff; skipping."
            )

    def test_jvp_matches_finite_diff_directional(self):
        # Everything in float64
        primals = (
            self.Q.clone(),
            self.m.clone(),
            self.c.clone(),
            self.X.clone(),
            self.budget.clone(),
            self.kappa.clone(),
            self.theta.clone(),
            self.p.clone(),
            self.x0.clone(),
        )

        # Small tangents in double
        dQ = torch.zeros_like(self.Q)
        dQ[:, torch.arange(self.n), torch.arange(self.n)] = 1e-5
        dm = torch.full_like(self.m, 1e-5)
        dc = torch.full_like(self.c, 1e-5)
        dX = torch.full_like(self.X, 1e-5)
        db = torch.full_like(self.budget, 1e-5)
        dk = torch.full_like(self.kappa, 5e-6)
        dt = torch.full_like(self.theta, -4e-6)
        dp = torch.full_like(self.p, 1e-5)
        dx0 = torch.zeros_like(self.x0)  # treated as non-diff

        tangents = (dQ, dm, dc, dX, db, dk, dt, dp, dx0)

        def f(*args):
            return StockSolverFunc.apply(*args)

        x_base, dx = jvp(f, primals, tangents)

        eps = self.eps_fd
        primals_plus = tuple(t + eps * dt for t, dt in zip(primals, tangents))
        primals_minus = tuple(t - eps * dt for t, dt in zip(primals, tangents))
        x_plus = f(*primals_plus)
        x_minus = f(*primals_minus)
        dx_fd = (x_plus - x_minus) / (2 * eps)

        self.assertTrue(
            torch.isfinite(dx).all() and torch.isfinite(dx_fd).all(),
            "Non-finite JVP or FD directional derivative",
        )
        self.assertTrue(
            torch.allclose(dx, dx_fd, atol=self.atol, rtol=self.rtol),
            f"JVP mismatch (max abs diff={(dx - dx_fd).abs().max().item():.4e})",
        )

    def test_adjoint_directional_consistency_for_p(self):
        # ⟨v, J_p·u⟩ == ⟨(J_p)^T v, u⟩ in double
        Q, m, c, X = (
            self.Q.clone(),
            self.m.clone(),
            self.c.clone(),
            self.X.clone(),
        )
        budget, kappa, theta = (
            self.budget.clone(),
            self.kappa.clone(),
            self.theta.clone(),
        )
        p, x0 = self.p.clone(), self.x0.clone()

        u = (
            torch.tensor(
                [0.01, -0.02, 0.015], dtype=self.dtype, device=self.device
            )
            * 1e-2
        )

        def f(*args):
            return StockSolverFunc.apply(*args)

        x = f(Q, m, c, X, budget, kappa, theta, p, x0)
        v = (torch.randn_like(x)) * 1e-2

        primals = (Q, m, c, X, budget, kappa, theta, p, x0)
        tangents = (
            torch.zeros_like(Q),
            torch.zeros_like(m),
            torch.zeros_like(c),
            torch.zeros_like(X),
            torch.zeros_like(budget),
            torch.zeros_like(kappa),
            torch.zeros_like(theta),
            u,
            torch.zeros_like(x0),
        )
        _, dx = jvp(f, primals, tangents)
        lhs = (v * dx).sum()

        p_ = p.clone().requires_grad_(True)
        x2 = f(Q, m, c, X, budget, kappa, theta, p_, x0)
        (v.detach() * x2).sum().backward()
        rhs = (p_.grad * u).sum()

        self.assertTrue(
            torch.allclose(lhs, rhs, atol=5e-7, rtol=5e-7),
            f"Adjoint-directional consistency failed for p: "
            f"|lhs-rhs|={(lhs-rhs).abs().item():.3e}",
        )

    def test_adjoint_directional_consistency_for_X(self):
        # ⟨v, J_X·U⟩ == ⟨(J_X)^T v, U⟩ in double
        Q, m, c, X = (
            self.Q.clone(),
            self.m.clone(),
            self.c.clone(),
            self.X.clone(),
        )
        budget, kappa, theta = (
            self.budget.clone(),
            self.kappa.clone(),
            self.theta.clone(),
        )
        p, x0 = self.p.clone(), self.x0.clone()

        U_dir = (
            torch.tensor(
                [[0.01, 0.02, -0.01], [-0.015, 0.01, 0.005]],
                dtype=self.dtype,
                device=self.device,
            )
            * 1e-2
        )

        def f(*args):
            return StockSolverFunc.apply(*args)

        x = f(Q, m, c, X, budget, kappa, theta, p, x0)
        v = (torch.randn_like(x)) * 1e-2

        primals = (Q, m, c, X, budget, kappa, theta, p, x0)
        tangents = (
            torch.zeros_like(Q),
            torch.zeros_like(m),
            torch.zeros_like(c),
            U_dir,
            torch.zeros_like(budget),
            torch.zeros_like(kappa),
            torch.zeros_like(theta),
            torch.zeros_like(p),
            torch.zeros_like(x0),
        )
        _, dx = jvp(f, primals, tangents)
        lhs = (v * dx).sum()

        X_ = X.clone().requires_grad_(True)
        x2 = f(Q, m, c, X_, budget, kappa, theta, p, x0)
        (v.detach() * x2).sum().backward()
        rhs = (X_.grad * U_dir).sum()

        self.assertTrue(
            torch.allclose(lhs, rhs, atol=5e-7, rtol=5e-7),
            f"Adjoint-directional consistency failed for X: "
            f"|lhs-rhs|={(lhs-rhs).abs().item():.3e}",
        )

def _convert_price(
    Q: torch.Tensor,
    m: torch.Tensor,
    budget: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    p: torch.Tensor,
):
    p0 = p.repeat(m.shape[0], 1)
    m1 = -m + p0 + 2.0
    L = -1.0 / p0 * ((budget * kappa).unsqueeze(0))
    U = 1.0 / p0 * ((budget * theta).unsqueeze(0))
    # because of sign, theta and kappa are swapped, this is not an error!
    # this is cash restriction !
    # replace self.theta, self.kappa to change cash budget multiplier
    wl = budget * (1.0 - theta)
    wh = budget * (1.0 + kappa)
    # validate starting point and update if needed
    return p0, m1, wl, wh, L, U


def stock_solver_func(
    Q: torch.Tensor,  # (B, n, n)
    M: torch.Tensor,  # (B, n)
    budget: torch.Tensor,  # (B, )
    kappa: torch.Tensor,  # (B, )
    theta: torch.Tensor,  # (B, )
    p: torch.Tensor,  # (n)
    x0: torch.Tensor,  # (B, n)
) -> Tuple[Tensor, ...] | Tensor:
    return StockSolverFunc.apply(
        Q,
        M,
        torch.zeros_like(kappa),
        torch.zeros_like(M),
        budget,
        kappa,
        theta,
        p,
        x0,
    )


def stock_solver_func_ref(
    Q: torch.Tensor,  # (B, n, n)
    M: torch.Tensor,  # (B, n)
    budget: torch.Tensor,  # (B, )
    kappa: torch.Tensor,  # (B, )
    theta: torch.Tensor,  # (B, )
    p: torch.Tensor,  # (n)
    x0: torch.Tensor,  # (B, n)
) -> Tuple[Tensor, ...] | Tensor:
    p0, m1, wl, wh, L, U, x0 = StockSolverFunc._convert_price(
        Q,
        M,
        torch.zeros_like(kappa),
        torch.zeros_like(M),
        budget,
        kappa,
        theta,
        p,
        torch.zeros_like(M),
    )
    # x0 = find_start_point(wl, wh, L, U, p0)
    return ActiveSetQPFunc.apply(
        Q, m1, torch.zeros_like(kappa), wl, wh, L, U, p0, x0
    )[0]


cov_list = [
    np.array(
        [
            [
                1.4967844158707652e-08,
                4.4861219412749655e-09,
                4.7674373705712905e-09,
                6.5056385388013297e-09,
                8.4093593907286485e-09,
                1.0864468891359620e-08,
                1.3507586960508894e-08,
                1.4293700297668268e-08,
                1.3958507215595206e-08,
                1.1425033288240738e-08,
                9.2918154924709430e-09,
            ],
            [
                4.4861219412749655e-09,
                2.1222312665908517e-08,
                1.2215050469402262e-08,
                1.8282380047004995e-08,
                2.8055952019216757e-08,
                3.9060202621599349e-08,
                5.2586031522043364e-08,
                5.7581660389504861e-08,
                5.8927748044493346e-08,
                5.3595171315160905e-08,
                4.8460209307698160e-08,
            ],
            [
                4.7674373705712905e-09,
                1.2215050469402262e-08,
                3.5184713199883376e-08,
                4.1206883569020752e-08,
                7.4231556551353932e-08,
                1.0664169754235178e-07,
                1.4986840524534028e-07,
                1.7075540444467941e-07,
                1.8093073216752925e-07,
                1.7126016310583920e-07,
                1.4702304134208357e-07,
            ],
            [
                6.5056385388013297e-09,
                1.8282380047004995e-08,
                4.1206883569020752e-08,
                1.1976751858165693e-07,
                2.2624337663931502e-07,
                3.2946546766212986e-07,
                4.7789891930767954e-07,
                5.5838196394855093e-07,
                5.9635735649182673e-07,
                5.6447717810846433e-07,
                4.5999302427393140e-07,
            ],
            [
                8.4093593907286485e-09,
                2.8055952019216757e-08,
                7.4231556551353932e-08,
                2.2624337663931502e-07,
                7.3278620414908768e-07,
                1.0672231175048338e-06,
                1.6341789823363271e-06,
                1.9651705239192109e-06,
                2.1425244837474001e-06,
                2.1010631214283230e-06,
                1.7139311685832595e-06,
            ],
            [
                1.0864468891359620e-08,
                3.9060202621599349e-08,
                1.0664169754235178e-07,
                3.2946546766212986e-07,
                1.0672231175048338e-06,
                1.7705959073170719e-06,
                2.7491988231348976e-06,
                3.3975899295838840e-06,
                3.7880926211353724e-06,
                3.8803080847569443e-06,
                3.2685566439874434e-06,
            ],
            [
                1.3507586960508894e-08,
                5.2586031522043364e-08,
                1.4986840524534028e-07,
                4.7789891930767954e-07,
                1.6341789823363271e-06,
                2.7491988231348976e-06,
                4.8809936929360704e-06,
                6.2196280521879010e-06,
                7.1779829759745820e-06,
                7.8612584260827384e-06,
                6.8862684545245249e-06,
            ],
            [
                1.4293700297668268e-08,
                5.7581660389504861e-08,
                1.7075540444467941e-07,
                5.5838196394855093e-07,
                1.9651705239192109e-06,
                3.3975899295838840e-06,
                6.2196280521879010e-06,
                8.5178449084659586e-06,
                9.9938871730060835e-06,
                1.1585000383146240e-05,
                1.0511369243808631e-05,
            ],
            [
                1.3958507215595206e-08,
                5.8927748044493346e-08,
                1.8093073216752925e-07,
                5.9635735649182673e-07,
                2.1425244837474001e-06,
                3.7880926211353724e-06,
                7.1779829759745820e-06,
                9.9938871730060835e-06,
                1.2645863004098321e-05,
                1.5293313207202050e-05,
                1.4451199452069313e-05,
            ],
            [
                1.1425033288240738e-08,
                5.3595171315160905e-08,
                1.7126016310583920e-07,
                5.6447717810846433e-07,
                2.1010631214283230e-06,
                3.8803080847569443e-06,
                7.8612584260827384e-06,
                1.1585000383146240e-05,
                1.5293313207202050e-05,
                2.2120060833591408e-05,
                2.1743961405800702e-05,
            ],
            [
                9.2918154924709430e-09,
                4.8460209307698160e-08,
                1.4702304134208357e-07,
                4.5999302427393140e-07,
                1.7139311685832595e-06,
                3.2685566439874434e-06,
                6.8862684545245249e-06,
                1.0511369243808631e-05,
                1.4451199452069313e-05,
                2.1743961405800702e-05,
                2.5164598256124499e-05,
            ],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [
                1.1404655189070473e-06,
                1.0719011389188528e-06,
                1.0411255965945843e-06,
                9.9628145527254646e-07,
                8.8748660182922055e-07,
                8.4332251158565103e-07,
                6.2356272965650756e-07,
                4.5548407689138837e-07,
                2.6160670430182535e-07,
                4.6036929319699964e-08,
                -1.2997459386391059e-07,
            ],
            [
                1.0719011389188528e-06,
                1.2499645744702610e-06,
                1.2046601389403780e-06,
                1.2943112623199832e-06,
                1.3887880721484934e-06,
                1.5445222677559631e-06,
                1.5560160501195983e-06,
                1.4577332134584792e-06,
                1.3093404827551587e-06,
                1.0464409625364844e-06,
                8.4930876648518905e-07,
            ],
            [
                1.0411255965945843e-06,
                1.2046601389403780e-06,
                1.5102410496447681e-06,
                1.8259395578699697e-06,
                2.4744115918292764e-06,
                3.1398677442474743e-06,
                3.8973774051145936e-06,
                4.2148491303590793e-06,
                4.3924397605661880e-06,
                4.1499803048820643e-06,
                3.7352629963689702e-06,
            ],
            [
                9.9628145527254646e-07,
                1.2943112623199832e-06,
                1.8259395578699697e-06,
                3.2299246880209086e-06,
                5.5294364180068895e-06,
                7.7707583011788786e-06,
                1.0691976493235408e-05,
                1.2301918937841746e-05,
                1.3308000141193037e-05,
                1.2964163329788982e-05,
                1.1460157013807128e-05,
            ],
            [
                8.8748660182922055e-07,
                1.3887880721484934e-06,
                2.4744115918292764e-06,
                5.5294364180068895e-06,
                1.3876372478025126e-05,
                2.1155526248509003e-05,
                3.1580570071422121e-05,
                3.7504676183282630e-05,
                4.0946583915609370e-05,
                4.0174970469648238e-05,
                3.3416617682380016e-05,
            ],
            [
                8.4332251158565103e-07,
                1.5445222677559631e-06,
                3.1398677442474743e-06,
                7.7707583011788786e-06,
                2.1155526248509003e-05,
                3.4581429339059862e-05,
                5.4066306807594413e-05,
                6.6010460670165520e-05,
                7.3631114615820700e-05,
                7.4681257277656263e-05,
                6.2552646495916541e-05,
            ],
            [
                6.2356272965650756e-07,
                1.5560160501195983e-06,
                3.8973774051145936e-06,
                1.0691976493235408e-05,
                3.1580570071422121e-05,
                5.4066306807594413e-05,
                9.4174991241394680e-05,
                1.2024061461800940e-04,
                1.3955018675540620e-04,
                1.5013522827608965e-04,
                1.2851798857000627e-04,
            ],
            [
                4.5548407689138837e-07,
                1.4577332134584792e-06,
                4.2148491303590793e-06,
                1.2301918937841746e-05,
                3.7504676183282630e-05,
                6.6010460670165520e-05,
                1.2024061461800940e-04,
                1.6123878306054489e-04,
                1.9239211273963013e-04,
                2.1807534074840850e-04,
                1.9240065388790873e-04,
            ],
            [
                2.6160670430182535e-07,
                1.3093404827551587e-06,
                4.3924397605661880e-06,
                1.3308000141193037e-05,
                4.0946583915609370e-05,
                7.3631114615820700e-05,
                1.3955018675540620e-04,
                1.9239211273963013e-04,
                2.4420266028157817e-04,
                2.9317185333824669e-04,
                2.7060147483096251e-04,
            ],
            [
                4.6036929319699964e-08,
                1.0464409625364844e-06,
                4.1499803048820643e-06,
                1.2964163329788982e-05,
                4.0174970469648238e-05,
                7.4681257277656263e-05,
                1.5013522827608965e-04,
                2.1807534074840850e-04,
                2.9317185333824669e-04,
                4.0511552952814258e-04,
                3.9503235624246459e-04,
            ],
            [
                -1.2997459386391059e-07,
                8.4930876648518905e-07,
                3.7352629963689702e-06,
                1.1460157013807128e-05,
                3.3416617682380016e-05,
                6.2552646495916541e-05,
                1.2851798857000627e-04,
                1.9240065388790873e-04,
                2.7060147483096251e-04,
                3.9503235624246459e-04,
                4.2985742086697638e-04,
            ],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [
                2.3413225316649776e-06,
                4.3133533827263339e-06,
                4.2388449581853857e-06,
                4.0095785631264095e-06,
                3.5968563038507728e-06,
                3.4315049394992208e-06,
                2.5441916223357431e-06,
                1.9296695578757291e-06,
                1.0566863480082160e-06,
                2.7586951351536870e-07,
                -2.4200842331248970e-07,
            ],
            [
                4.3133533827263339e-06,
                9.1510586676011855e-06,
                8.9214723548541204e-06,
                8.6539332884051612e-06,
                8.1689159066132787e-06,
                8.2000646739824232e-06,
                6.7833910381965278e-06,
                5.7249354490805539e-06,
                4.0610735695600303e-06,
                2.4575649507100238e-06,
                1.8209080126865768e-06,
            ],
            [
                4.2388449581853857e-06,
                8.9214723548541204e-06,
                9.8264525154309954e-06,
                1.0509612377381843e-05,
                1.2087758062098330e-05,
                1.3935287285653831e-05,
                1.4916065947443860e-05,
                1.5183488914906641e-05,
                1.4554027880265137e-05,
                1.3232913199296753e-05,
                1.3162823047273402e-05,
            ],
            [
                4.0095785631264095e-06,
                8.6539332884051612e-06,
                1.0509612377381843e-05,
                1.4871510838565446e-05,
                2.2440014237664845e-05,
                2.9690826983403715e-05,
                3.7909474931939153e-05,
                4.2416830245258829e-05,
                4.4871372963030683e-05,
                4.4118333268445772e-05,
                4.2107510950656495e-05,
            ],
            [
                3.5968563038507728e-06,
                8.1689159066132787e-06,
                1.2087758062098330e-05,
                2.2440014237664845e-05,
                4.7859255657890701e-05,
                7.0968467842158638e-05,
                1.0175097650117948e-04,
                1.1961028034919704e-04,
                1.3092643675054056e-04,
                1.3148942947021808e-04,
                1.1745979695984476e-04,
            ],
            [
                3.4315049394992208e-06,
                8.2000646739824232e-06,
                1.3935287285653831e-05,
                2.9690826983403715e-05,
                7.0968467842158638e-05,
                1.1254804222787454e-04,
                1.7099436565334535e-04,
                2.0700702887171819e-04,
                2.3142536881584519e-04,
                2.3892176577222085e-04,
                2.1019841762837677e-04,
            ],
            [
                2.5441916223357431e-06,
                6.7833910381965278e-06,
                1.4916065947443860e-05,
                3.7909474931939153e-05,
                1.0175097650117948e-04,
                1.7099436565334535e-04,
                2.9187693502531471e-04,
                3.7337348682205145e-04,
                4.3945166703096745e-04,
                4.8233389149181455e-04,
                4.3003153435607446e-04,
            ],
            [
                1.9296695578757291e-06,
                5.7249354490805539e-06,
                1.5183488914906641e-05,
                4.2416830245258829e-05,
                1.1961028034919704e-04,
                2.0700702887171819e-04,
                3.7337348682205145e-04,
                4.9831805287104867e-04,
                6.0463716432872380e-04,
                6.9623358173297059e-04,
                6.3453881223751935e-04,
            ],
            [
                1.0566863480082160e-06,
                4.0610735695600303e-06,
                1.4554027880265137e-05,
                4.4871372963030683e-05,
                1.3092643675054056e-04,
                2.3142536881584519e-04,
                4.3945166703096745e-04,
                6.0463716432872380e-04,
                7.6824801296692855e-04,
                9.3041272669921321e-04,
                8.6968935276933066e-04,
            ],
            [
                2.7586951351536870e-07,
                2.4575649507100238e-06,
                1.3232913199296753e-05,
                4.4118333268445772e-05,
                1.3148942947021808e-04,
                2.3892176577222085e-04,
                4.8233389149181455e-04,
                6.9623358173297059e-04,
                9.3041272669921321e-04,
                1.2684020421648446e-03,
                1.2396643219062884e-03,
            ],
            [
                -2.4200842331248970e-07,
                1.8209080126865768e-06,
                1.3162823047273402e-05,
                4.2107510950656495e-05,
                1.1745979695984476e-04,
                2.1019841762837677e-04,
                4.3003153435607446e-04,
                6.3453881223751935e-04,
                8.6968935276933066e-04,
                1.2396643219062884e-03,
                1.3215357344108685e-03,
            ],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [
                2.4040995745840317e-06,
                6.2590988626813095e-06,
                8.6241469085818088e-06,
                8.3197576966715313e-06,
                7.7260021657291255e-06,
                7.5755433315680977e-06,
                6.2678647435877005e-06,
                5.2848737507178239e-06,
                3.8004301761305009e-06,
                1.9844998623709639e-06,
                1.2137251792592267e-06,
            ],
            [
                6.2590988626813095e-06,
                1.9473391215790647e-05,
                2.6383803117656981e-05,
                2.5178104866552967e-05,
                2.3062568820866322e-05,
                2.2488425104786989e-05,
                1.8366578080394039e-05,
                1.5273059680813589e-05,
                1.0675432833011227e-05,
                4.7328274552407496e-06,
                2.2596575869527221e-06,
            ],
            [
                8.6241469085818088e-06,
                2.6383803117656981e-05,
                3.7441912919126087e-05,
                3.7200963611408016e-05,
                3.7389780791504209e-05,
                3.9218411579455551e-05,
                3.6883181544131655e-05,
                3.4136825925393107e-05,
                2.9060309638313856e-05,
                2.0040131082045041e-05,
                1.7788901658223684e-05,
            ],
            [
                8.3197576966715313e-06,
                2.5178104866552967e-05,
                3.7200963611408016e-05,
                4.5689624834110487e-05,
                6.0537334773274511e-05,
                7.4728779892074091e-05,
                8.8766931519810246e-05,
                9.4477921188538614e-05,
                9.5942436652897336e-05,
                8.8549274454032562e-05,
                8.4975501477948094e-05,
            ],
            [
                7.7260021657291255e-06,
                2.3062568820866322e-05,
                3.7389780791504209e-05,
                6.0537334773274511e-05,
                1.1341569984523452e-04,
                1.6187591387863186e-04,
                2.2400854036686574e-04,
                2.5738398527287042e-04,
                2.7867241900985377e-04,
                2.7946061527072915e-04,
                2.5705256597011113e-04,
            ],
            [
                7.5755433315680977e-06,
                2.2488425104786989e-05,
                3.9218411579455551e-05,
                7.4728779892074091e-05,
                1.6187591387863186e-04,
                2.4885349768843728e-04,
                3.6840616829545932e-04,
                4.3836029131145965e-04,
                4.8631684920695393e-04,
                5.0458495883407700e-04,
                4.5728076639480869e-04,
            ],
            [
                6.2678647435877005e-06,
                1.8366578080394039e-05,
                3.6883181544131655e-05,
                8.8766931519810246e-05,
                2.2400854036686574e-04,
                3.6840616829545932e-04,
                6.0876143121090355e-04,
                7.6788862338952435e-04,
                8.9507491808325725e-04,
                9.9093575928838116e-04,
                9.0752360075895217e-04,
            ],
            [
                5.2848737507178239e-06,
                1.5273059680813589e-05,
                3.4136825925393107e-05,
                9.4477921188538614e-05,
                2.5738398527287042e-04,
                4.3836029131145965e-04,
                7.6788862338952435e-04,
                1.0112246229861373e-03,
                1.2169464264570723e-03,
                1.4189927126184766e-03,
                1.3276891030699059e-03,
            ],
            [
                3.8004301761305009e-06,
                1.0675432833011227e-05,
                2.9060309638313856e-05,
                9.5942436652897336e-05,
                2.7867241900985377e-04,
                4.8631684920695393e-04,
                8.9507491808325725e-04,
                1.2169464264570723e-03,
                1.5302573198485768e-03,
                1.8792317306918272e-03,
                1.7952278842611833e-03,
            ],
            [
                1.9844998623709639e-06,
                4.7328274552407496e-06,
                2.0040131082045041e-05,
                8.8549274454032562e-05,
                2.7946061527072915e-04,
                5.0458495883407700e-04,
                9.9093575928838116e-04,
                1.4189927126184766e-03,
                1.8792317306918272e-03,
                2.5835797097870676e-03,
                2.5542232254152069e-03,
            ],
            [
                1.2137251792592267e-06,
                2.2596575869527221e-06,
                1.7788901658223684e-05,
                8.4975501477948094e-05,
                2.5705256597011113e-04,
                4.5728076639480869e-04,
                9.0752360075895217e-04,
                1.3276891030699059e-03,
                1.7952278842611833e-03,
                2.5542232254152069e-03,
                2.7348447716831060e-03,
            ],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [
                2.5051367615385421e-06,
                5.8787874951866763e-06,
                1.1622605810365318e-05,
                1.5739822324186890e-05,
                1.5423624747189841e-05,
                1.5583107203054783e-05,
                1.4001420632559547e-05,
                1.2369925478240125e-05,
                9.8281016278836460e-06,
                5.2876941060228621e-06,
                3.3480863568081205e-06,
            ],
            [
                5.8787874951866763e-06,
                1.8615524599746053e-05,
                3.5816215765153280e-05,
                4.8272551211917176e-05,
                4.6495436472278934e-05,
                4.6545675832698062e-05,
                4.1082125239078437e-05,
                3.5649511643258641e-05,
                2.7706431468876417e-05,
                1.3445438907271958e-05,
                7.0922572186406071e-06,
            ],
            [
                1.1622605810365318e-05,
                3.5816215765153280e-05,
                7.2121987055907111e-05,
                9.6332891681587046e-05,
                9.1979973480266959e-05,
                9.1752364586764545e-05,
                8.0297519995515946e-05,
                6.8670804675522535e-05,
                5.2825912330884630e-05,
                2.3017340773498344e-05,
                8.4752958730443326e-06,
            ],
            [
                1.5739822324186890e-05,
                4.8272551211917176e-05,
                9.6332891681587046e-05,
                1.3807520008331893e-04,
                1.5464104542889387e-04,
                1.7214035084247100e-04,
                1.8053865495425424e-04,
                1.7547020230830878e-04,
                1.6324782242823691e-04,
                1.2387772105811802e-04,
                9.9033264185672035e-05,
            ],
            [
                1.5423624747189841e-05,
                4.6495436472278934e-05,
                9.1979973480266959e-05,
                1.5464104542889387e-04,
                2.5509035020591047e-04,
                3.4183607780513612e-04,
                4.4822664644377723e-04,
                4.9361730431600785e-04,
                5.2036401047795178e-04,
                5.0355164820445721e-04,
                4.3500615795596297e-04,
            ],
            [
                1.5583107203054783e-05,
                4.6545675832698062e-05,
                9.1752364586764545e-05,
                1.7214035084247100e-04,
                3.4183607780513612e-04,
                4.9938251277044008e-04,
                7.1207017180784191e-04,
                8.2049636362772373e-04,
                8.9406741837344604e-04,
                9.1657573251558236e-04,
                7.9278095509890544e-04,
            ],
            [
                1.4001420632559547e-05,
                4.1082125239078437e-05,
                8.0297519995515946e-05,
                1.8053865495425424e-04,
                4.4822664644377723e-04,
                7.1207017180784191e-04,
                1.1416923450289042e-03,
                1.4105282248850358e-03,
                1.6262335704057540e-03,
                1.8285387230134692e-03,
                1.6294907902617605e-03,
            ],
            [
                1.2369925478240125e-05,
                3.5649511643258641e-05,
                6.8670804675522535e-05,
                1.7547020230830878e-04,
                4.9361730431600785e-04,
                8.2049636362772373e-04,
                1.4105282248850358e-03,
                1.8329861019303101e-03,
                2.1962262952575505e-03,
                2.6274403890247039e-03,
                2.4114640838847948e-03,
            ],
            [
                9.8281016278836460e-06,
                2.7706431468876417e-05,
                5.2825912330884630e-05,
                1.6324782242823691e-04,
                5.2036401047795178e-04,
                8.9406741837344604e-04,
                1.6262335704057540e-03,
                2.1962262952575505e-03,
                2.7577307758779035e-03,
                3.4975481565237038e-03,
                3.3072100010525206e-03,
            ],
            [
                5.2876941060228621e-06,
                1.3445438907271958e-05,
                2.3017340773498344e-05,
                1.2387772105811802e-04,
                5.0355164820445721e-04,
                9.1657573251558236e-04,
                1.8285387230134692e-03,
                2.6274403890247039e-03,
                3.4975481565237038e-03,
                4.9298839854205077e-03,
                4.8309921618852718e-03,
            ],
            [
                3.3480863568081205e-06,
                7.0922572186406071e-06,
                8.4752958730443326e-06,
                9.9033264185672035e-05,
                4.3500615795596297e-04,
                7.9278095509890544e-04,
                1.6294907902617605e-03,
                2.4114640838847948e-03,
                3.3072100010525206e-03,
                4.8309921618852718e-03,
                5.0923445181415309e-03,
            ],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [
                2.7707272100894262e-06,
                5.0086056985205372e-06,
                1.0020202022549428e-05,
                1.9131165070400176e-05,
                2.7657950706786207e-05,
                3.1100837630338745e-05,
                3.2933835193982148e-05,
                3.0626222999306671e-05,
                2.7054291115824083e-05,
                1.5471468929660596e-05,
                1.3594144575560041e-05,
            ],
            [
                5.0086056985205372e-06,
                1.6658497834927466e-05,
                3.1178828718253667e-05,
                5.9696154661413551e-05,
                8.5298945159150051e-05,
                9.4733162722930741e-05,
                9.8673751837397334e-05,
                9.0560374404331662e-05,
                7.8631017776795389e-05,
                4.2684709408874384e-05,
                3.6025853652567215e-05,
            ],
            [
                1.0020202022549428e-05,
                3.1178828718253667e-05,
                6.4181237402783007e-05,
                1.2127095672420158e-04,
                1.7249720113685400e-04,
                1.9030449679691673e-04,
                1.9633683041464837e-04,
                1.7829530423429829e-04,
                1.5273948520557362e-04,
                7.9408263073034192e-05,
                6.2729378644777793e-05,
            ],
            [
                1.9131165070400176e-05,
                5.9696154661413551e-05,
                1.2127095672420158e-04,
                2.3629828882698470e-04,
                3.3635972597579364e-04,
                3.7196214339258994e-04,
                3.8641761187623019e-04,
                3.5249594703983932e-04,
                3.0279852243293644e-04,
                1.6328669574239843e-04,
                1.2034408932520100e-04,
            ],
            [
                2.7657950706786207e-05,
                8.5298945159150051e-05,
                1.7249720113685400e-04,
                3.3635972597579364e-04,
                5.8080344845630461e-04,
                7.6098938336145315e-04,
                9.6283283339454711e-04,
                1.0031806653658232e-03,
                9.9795067957116372e-04,
                8.4616189073818851e-04,
                7.2018158715910765e-04,
            ],
            [
                3.1100837630338745e-05,
                9.4733162722930741e-05,
                1.9030449679691673e-04,
                3.7196214339258994e-04,
                7.6098938336145315e-04,
                1.1357734236982304e-03,
                1.6158572490060747e-03,
                1.7998119840358956e-03,
                1.8971861790073733e-03,
                1.8213639170328322e-03,
                1.5717133302137248e-03,
            ],
            [
                3.2933835193982148e-05,
                9.8673751837397334e-05,
                1.9633683041464837e-04,
                3.8641761187623019e-04,
                9.6283283339454711e-04,
                1.6158572490060747e-03,
                2.5961383922288323e-03,
                3.1094102617080220e-03,
                3.4761040070747751e-03,
                3.7047131502558131e-03,
                3.2430075863150490e-03,
            ],
            [
                3.0626222999306671e-05,
                9.0560374404331662e-05,
                1.7829530423429829e-04,
                3.5249594703983932e-04,
                1.0031806653658232e-03,
                1.7998119840358956e-03,
                3.1094102617080220e-03,
                3.9268132737901636e-03,
                4.5828705320687989e-03,
                5.2373840418409711e-03,
                4.7045235876457848e-03,
            ],
            [
                2.7054291115824083e-05,
                7.8631017776795389e-05,
                1.5273948520557362e-04,
                3.0279852243293644e-04,
                9.9795067957116372e-04,
                1.8971861790073733e-03,
                3.4761040070747751e-03,
                4.5828705320687989e-03,
                5.6065675388574333e-03,
                6.8497747171097122e-03,
                6.3632919890464481e-03,
            ],
            [
                1.5471468929660596e-05,
                4.2684709408874384e-05,
                7.9408263073034192e-05,
                1.6328669574239843e-04,
                8.4616189073818851e-04,
                1.8213639170328322e-03,
                3.7047131502558131e-03,
                5.2373840418409711e-03,
                6.8497747171097122e-03,
                9.3448156145964495e-03,
                8.9881913463269406e-03,
            ],
            [
                1.3594144575560041e-05,
                3.6025853652567215e-05,
                6.2729378644777793e-05,
                1.2034408932520100e-04,
                7.2018158715910765e-04,
                1.5717133302137248e-03,
                3.2430075863150490e-03,
                4.7045235876457848e-03,
                6.3632919890464481e-03,
                8.9881913463269406e-03,
                9.2798435729837744e-03,
            ],
        ],
        dtype=np.float64,
    ),
]

mean_list = [
    np.array(
        [
            4.2915767955818578e-05,
            4.5552443170794496e-05,
            4.8264989935137231e-05,
            5.0294615685219668e-05,
            5.8857112001185896e-05,
            6.7577235678571731e-05,
            8.2596648354498934e-05,
            8.5259937283451378e-05,
            8.0152799987454632e-05,
            5.9541828977602564e-05,
            3.5317750823330857e-05,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.0008885068792790043,
            0.0009465000894787689,
            0.0010176058074618855,
            0.001047175929787638,
            0.0012282943418884238,
            0.0014108335297061726,
            0.0017331969464183812,
            0.0017972476224603206,
            0.0016812042287640083,
            0.0012722154125145136,
            0.0007605863383324029,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.0012514748988301527,
            0.002676020570220902,
            0.0029345640283889303,
            0.002991081961792117,
            0.0035043577711223626,
            0.004019834319509856,
            0.004972320452406881,
            0.0052063323885976134,
            0.004877068580714551,
            0.003767218361757748,
            0.0023096304818408762,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.0012293831006866612,
            0.00387425660560433,
            0.005814035174241505,
            0.006046535310213,
            0.0070857144900155055,
            0.008160147191190957,
            0.0102260381268029,
            0.010865809907244653,
            0.010214995036612797,
            0.007963192824888238,
            0.0048797078090217424,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.0011799048365380104,
            0.003730492787580955,
            0.007933349918606438,
            0.01172203706712558,
            0.013443535586294277,
            0.015692970145663315,
            0.020153350617705238,
            0.021966342230513063,
            0.020782713775002237,
            0.01641070799467345,
            0.010042399360325803,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.0010753434296864026,
            0.0034243643349487632,
            0.00736239510150649,
            0.015669934110612956,
            0.024498156315578393,
            0.028559505749387395,
            0.037591678439858785,
            0.04245940613068935,
            0.040855980639036275,
            0.03312685313134875,
            0.02124521348306663,
        ],
        dtype=np.float64,
    ),
]

class TestQPStockSoverGrad(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    @staticmethod
    def _model_agents_numpy(B, M_ave, Sigma_ave, seed=None):
        """
        NumPy/SciPy rewrite of model_agents:
          - NumPy for arrays & linear algebra
          - SciPy.stats for random sampling
        """
        rng = np.random.default_rng(seed)

        n = M_ave.shape[0]

        # Hyperparameters
        budget_b = 10.0
        budget_scale = 130000.0
        kappa_a = 1.0
        kappa_b = 4.0
        theta_a = 2.0
        theta_b = 2.0
        lam_a = 1.0
        lam_scale = 0.1  # note: SciPy's gamma uses 'scale' = 1/rate

        # Cholesky of the average covariance
        L_omega = np.linalg.cholesky(Sigma_ave)  # [n, n]

        # --- Priors ---
        # Multivariate normals (B draws)
        M = rng.multivariate_normal(
            mean=M_ave, cov=Sigma_ave, size=B
        )  # [B, n]

        # Half-Cauchy std devs per dimension per draw
        sigma1 = stats.halfcauchy.rvs(
            loc=0.0, scale=2.5, size=(B, n), random_state=rng
        )  # [B, n]

        # Build covariance Cholesky per draw: L_Sigma_b = diag(sigma1_b) @ L_omega
        # Equivalent to row-wise scaling of L_omega by sigma1
        L_Sigma = L_omega[None, :, :] * sigma1[:, :, None]  # [B, n, n]

        # Sigma_b = L_Sigma_b @ L_Sigma_b.T + 1e-3 * I
        Sigma = L_Sigma @ np.transpose(L_Sigma, (0, 2, 1)) + (
            np.eye(n)[None, :, :] * 1e-3
        )  # [B, n, n]

        # Pareto(scale=1.0, alpha=budget_b) then scale up
        budget = budget_scale * stats.pareto.rvs(
            b=budget_b, scale=1.0, size=B, random_state=rng
        )  # [B]

        # Betas
        kappa = stats.beta.rvs(
            a=kappa_a, b=kappa_b, size=B, random_state=rng
        )  # [B]
        theta_raw = stats.beta.rvs(
            a=theta_a, b=theta_b, size=B, random_state=rng
        )  # [B]
        theta = theta_raw * ((n - 1) / n) + (1 / n)  # map to [1/n, 1]

        return Sigma, M, budget, kappa, theta

    @staticmethod
    def _convert_price(
        Q: torch.Tensor,
        m: torch.Tensor,
        budget: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        p: torch.Tensor,
    ):
        p0 = p.repeat(m.shape[0], axis=1).T  # TODO: change to m.size(0)
        m1 = -m + p0 + 2.0
        L = -1.0 / p0 * ((budget * kappa)[:, None])
        U = 1.0 / p0 * ((budget * theta)[:, None])
        # because of sign, theta and kappa are swapped, this is not an error!
        # this is cash restriction !
        # replace self.theta, self.kappa to change cash budget multiplier
        wl = budget * (1.0 - theta)
        wh = budget * (1.0 + kappa)
        # validate starting point and update if needed
        return p0, m1, wl, wh, L, U

    @staticmethod
    def _find_start_point(
        p: torch.Tensor,  # (B, n)
        wl: torch.Tensor,  # (B,)
        wh: torch.Tensor,  # (B,)
        L: torch.Tensor,  # (B, n)
        U: torch.Tensor,  # (B, n)
    ) -> torch.Tensor:
        eps = 1e-8
        """
        Return x in R^{B x n} such that:
          L < x < U   (componentwise, up to numerical tolerance)
          wl < p^T x < wh
        assuming feasibility: p^T U > wl and p^T L < wh (per batch).

        The construction is: x = L + alpha * (U - L), for an alpha in (0,1) that
        also makes p^T x land strictly inside (wl, wh).
        """

        # assert p.shape == L.shape == U.shape, \
        #    f"Shapes must match: p {p.shape}, L {L.shape}, U {U.shape}"

        B, n = L.shape

        # Compute p^T L and p^T U (batchwise)
        p_dot_L = bdot(p, L)  # (B,)
        p_dot_U = bdot(p, U)  # (B,)
        den = p_dot_U - p_dot_L  # (B,)

        # Target alpha interval coming from wl < p^T(L + alpha(U-L)) < wh
        # a = (target - p^T L) / (p^T U - p^T L)
        a_l = (wl - p_dot_L) / den
        a_h = (wh - p_dot_L) / den

        # Order the endpoints to get [a_min, a_max]
        a_min = torch.minimum(a_l, a_h)
        a_max = torch.maximum(a_l, a_h)

        # Intersect with (0,1) and pull strictly inside by eps
        lo = torch.clamp(a_min, min=0.0, max=1.0)
        hi = torch.clamp(a_max, min=0.0, max=1.0)

        # nudge away from boundaries to keep strict inequalities
        lo = torch.clamp(lo, min=eps, max=1.0 - eps)
        hi = torch.clamp(hi, min=eps, max=1.0 - eps)

        # Handle degenerate case den ~ 0 (p^T is constant along the segment)
        # If p^T L == p^T U and it's already strictly inside (wl, wh),
        # pick alpha=0.5 (then only the box matters).
        den_is_small = den.abs() <= 1e-12
        inside_when_flat = (p_dot_L > wl) & (p_dot_L < wh) & den_is_small

        # For normal cases, midpoint of the feasible alpha interval
        alpha = 0.5 * (lo + hi)

        # Where den is ~0 but feasible, choose 0.5 (and keep it away from box edges)
        if inside_when_flat.any():
            alpha = alpha.clone()
            alpha[inside_when_flat] = 0.5
            # keep margin wrt box with eps (already in (0,1) by 0.5)

        # As a final guard (feasible problems should not hit this), clip to (eps,1-eps)
        alpha = torch.clamp(alpha, min=eps, max=1.0 - eps)

        # Construct x
        return L + alpha.unsqueeze(1) * (U - L)

    def _prepare_solver_problem(self, seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(mean_list))

        Sigma, M, budget, kappa, theta = self._model_agents_numpy(
            1, mean_list[idx], cov_list[idx], seed
        )

        p0, m1, wl, wh, L, U = self._convert_price(
            Sigma, M, budget, kappa, theta, np.full((M.shape[1], 1), 100)
        )
        x0 = self._find_start_point(
            torch.as_tensor(p0),
            torch.as_tensor(wl),
            torch.as_tensor(wh),
            torch.as_tensor(L),
            torch.as_tensor(U),
        ).numpy(),
        p0 = p0[0].copy()
        x0 = x0[0].copy()
        return Sigma, M, budget, kappa, theta, p0, x0

    def test_stock_solver_grad(self):
        for i in range(1000):
            Sigma, m1, budget, kappa, theta, p0, x0 = self._prepare_solver_problem(i)

            # Create tensors
            Sigma = torch.as_tensor(Sigma, dtype=torch.float64)
            m1 = torch.as_tensor(m1, dtype=torch.float64)
            budget = torch.as_tensor(budget, dtype=torch.float64)
            kappa = torch.as_tensor(kappa, dtype=torch.float64)
            theta = torch.as_tensor(theta, dtype=torch.float64)
            p0 = torch.as_tensor(p0, dtype=torch.float64)
            x0 = torch.as_tensor(x0, dtype=torch.float64)

            # Enable gradients on all inputs except x0
            Sigma.requires_grad_(True)
            m1.requires_grad_(True)
            budget.requires_grad_(True)
            kappa.requires_grad_(True)
            theta.requires_grad_(True)
            p0.requires_grad_(True)

            # Forward pass
            x1 = stock_solver_func(Sigma, m1, budget, kappa, theta, p0, x0)
            x2 = stock_solver_func_ref(Sigma, m1, budget, kappa, theta, p0, x0)

            self.assertTrue(torch.allclose(x1, x2, atol=1e-6, rtol=1e-6))

            # Prepare cloned inputs for the reference backward pass
            inputs_1 = [Sigma, m1, budget, kappa, theta, p0]
            inputs_2 = [t.detach().clone().requires_grad_(True) for t in inputs_1]
            # Backward for impl 1
            x1.sum().backward()


            # Backward for reference impl
            x2_ref = stock_solver_func_ref(
                inputs_2[0], inputs_2[1], inputs_2[2],
                inputs_2[3], inputs_2[4], inputs_2[5], x0
            )
            x2_ref.sum().backward()

            # Compare gradients
            names = ["Sigma", "m1", "budget", "kappa", "theta", "p0"]
            for g1, g2, name in zip(inputs_1, inputs_2, names):
                self.assertTrue(
                    torch.allclose(g1.grad, g2.grad, atol=1e-6, rtol=1e-6),
                    msg=f"Gradient mismatch for {name}"
                )


if __name__ == "__main__":
    unittest.main()
