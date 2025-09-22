import importlib
import unittest
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
from torch.autograd import grad

_core = importlib.import_module("heteromarket.core")
ActiveSetQPFunc = _core.ActiveSetQPFunc
bdot = _core.bdot


def find_start_point(wl, wh, L, U, p):
    denom = bdot(p, U).clamp_min(1e-30)
    x0 = torch.where(
        wl.unsqueeze(-1) < 0.0,
        torch.zeros_like(p),
        U * (wl / denom).unsqueeze(-1),
    )
    return x0


def solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0):
    r = ActiveSetQPFunc.apply(Q, m, c, wl, wh, L, U, p, x0)
    return r[0]

# Call autograd.Function in eager mode
def solve_QP_problem2(Q, m, c, wl, wh, L, U, p, x0):
    return ActiveSetQPFunc.apply(Q, m, c, wl, wh, L, U, p, x0)[0]


class TestSolveUnderActive(unittest.TestCase):
    def convert_active_set(
        self,
        active_low,
        active_high,
        active_zero,
        active_budg_low,
        active_budg_high,
        L,
        U,
        wl,
        wh,
    ):
        # ---- active mask ----
        active = active_low | active_high | active_zero  # (B, n)

        # Values to pin at active coords
        active_values = torch.zeros_like(L)
        active_values[active_low] = L[active_low]
        active_values[active_high] = U[active_high]
        # Not needed - already zero
        # active_values[active_zero] = 0.0

        budget_active = active_budg_low | active_budg_high  # (B,)
        budget_w = torch.where(active_budg_low, wl, wh)  # (B,)
        return active, active_values, budget_active, budget_w

    # Helper to build a small, well-conditioned SPD problem
    def make_small_problem(self, B=1, n=3, dtype=torch.float64, device="cpu"):
        torch.manual_seed(0)
        A = torch.randn(B, n, n, dtype=dtype, device=device)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(
            n, dtype=dtype, device=device
        )  # SPD
        m = torch.randn(B, n, dtype=dtype, device=device)

        # Simple bounds that include zero
        L = torch.full((B, n), -1.0, dtype=dtype, device=device)
        U = torch.full((B, n), 2.0, dtype=dtype, device=device)

        # Budget weights and targets
        p = torch.randn(B, n, dtype=dtype, device=device)
        wl = torch.full((B,), 0.25, dtype=dtype, device=device)
        wh = torch.full((B,), 0.75, dtype=dtype, device=device)

        # Commission thresholds (not used here but required by __init__)
        c = torch.full((B,), 0.1, dtype=dtype, device=device)
        return Q, m, p, c, wl, wh, L, U

    def test_no_actives_no_budget_matches_unconstrained(self):
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=1, n=3)
        B, n = Q.shape[0], Q.shape[1]
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.zeros(B, dtype=torch.bool)
        active_budg_high = torch.zeros(B, dtype=torch.bool)
        active, active_values, budget_active, budget_w = (
            self.convert_active_set(
                active_low,
                active_high,
                active_zero,
                active_budg_low,
                active_budg_high,
                L,
                U,
                wl,
                wh,
            )
        )
        x = ActiveSetQPFunc._solve_under_active(
            Q, m ,p, active, active_values, budget_active, budget_w
        )

        # Expected: solve H x = -m
        x_expected = torch.linalg.solve((2*Q)[0], -m[0]).unsqueeze(0)
        self.assertTrue(torch.allclose(x, x_expected, atol=1e-9, rtol=1e-9))

    def test_single_pinned_lower_bound_is_respected(self):
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=1, n=3)
        B, n = Q.shape[0], Q.shape[1]
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.zeros(B, dtype=torch.bool)
        active_budg_high = torch.zeros(B, dtype=torch.bool)

        # Pin coordinate 0 to its lower bound
        active_low[:, 0] = True

        active, active_values, budget_active, budget_w = (
            self.convert_active_set(
                active_low,
                active_high,
                active_zero,
                active_budg_low,
                active_budg_high,
                L,
                U,
                wl,
                wh,
            )
        )
        x = ActiveSetQPFunc._solve_under_active(
            Q, m, p, active, active_values, budget_active, budget_w
        )

        # Check pinned value
        self.assertTrue(torch.allclose(x[:, 0], L[:, 0], atol=1e-9))
        # Check free-block KKT: H_FF x_F = -(m_F + H_FB x_B)
        H = (2*Q)[0]
        m0 = m[0]
        B_idx = torch.tensor([0])
        F_idx = torch.tensor([1, 2])
        H_FF = H[F_idx][:, F_idx]
        H_FB = H[F_idx][:, B_idx]
        x_B = L[0, B_idx]
        rhs = -(m0[F_idx] + (H_FB @ x_B.unsqueeze(-1)).squeeze(-1))
        x_F_expected = torch.linalg.solve(H_FF, rhs)
        self.assertTrue(
            torch.allclose(x[0, F_idx], x_F_expected, atol=1e-9, rtol=1e-9)
        )

    def test_budget_low_projection_no_pins_hits_target_plane(self):
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=1, n=3)
        B, n = Q.shape[0], Q.shape[1]
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.tensor([True])
        active_budg_high = torch.tensor([False])

        active, active_values, budget_active, budget_w = (
            self.convert_active_set(
                active_low,
                active_high,
                active_zero,
                active_budg_low,
                active_budg_high,
                L,
                U,
                wl,
                wh,
            )
        )
        x = ActiveSetQPFunc._solve_under_active(
            Q, m, p, active, active_values, budget_active, budget_w
        )

        # Must satisfy p^T x = wl (within tolerance)
        p_dot_x = (p * x).sum(dim=-1)
        self.assertTrue(torch.allclose(p_dot_x, wl, atol=1e-9, rtol=1e-9))

        # Check projection formula x = x0 + H^{-1} p * (w - p^T x0) / (p^T H^{-1} p)
        H = (2*Q)[0]
        x0 = torch.linalg.solve(
            H, -m[0]
        )  # unconstrained under current active set (none)
        y = torch.linalg.solve(H, p[0])
        alpha = (wl[0] - (p[0] @ x0)) / (p[0] @ y)
        x_exp = x0 + alpha * y
        self.assertTrue(torch.allclose(x[0], x_exp, atol=1e-9, rtol=1e-9))

    def test_budget_projection_with_pinned_coord_does_not_move_pinned(self):
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=1, n=3)
        B, n = Q.shape[0], Q.shape[1]
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        # Pin index 1 to zero
        active_zero[:, 1] = True
        active_budg_low = torch.tensor([False])
        active_budg_high = torch.tensor([True])  # project to high plane

        active, active_values, budget_active, budget_w = (
            self.convert_active_set(
                active_low,
                active_high,
                active_zero,
                active_budg_low,
                active_budg_high,
                L,
                U,
                wl,
                wh,
            )
        )
        x = ActiveSetQPFunc._solve_under_active(
            Q, m, p, active, active_values, budget_active, budget_w
        )

        # Pinned coordinate unchanged (== 0)
        self.assertTrue(
            torch.allclose(x[:, 1], torch.zeros_like(x[:, 1]), atol=1e-12)
        )

        # Should satisfy p^T x = wh (within tolerance)
        p_dot_x = (p * x).sum(dim=-1)
        self.assertTrue(torch.allclose(p_dot_x, wh, atol=1e-9, rtol=1e-9))

    def test_batched_two_problems(self):
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=2, n=2)
        B, n = 2, 2

        # No pins in batch 0, pin x0 at L in batch 1
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_low[1, 0] = True

        # Budget: low active for batch 0, none for batch 1
        active_budg_low = torch.tensor([True, False], dtype=torch.bool)
        active_budg_high = torch.tensor([False, False], dtype=torch.bool)

        active, active_values, budget_active, budget_w = (
            self.convert_active_set(
                active_low,
                active_high,
                active_zero,
                active_budg_low,
                active_budg_high,
                L,
                U,
                wl,
                wh,
            )
        )
        x = ActiveSetQPFunc._solve_under_active(
            Q, m, p, active, active_values, budget_active, budget_w
        )

        # Batch 0: check budget equality
        p_dot_x0 = (p[0] * x[0]).sum()
        self.assertAlmostEqual(float(p_dot_x0), float(wl[0]), places=9)

        # Batch 1: check pin respected
        self.assertAlmostEqual(float(x[1, 0]), float(L[1, 0]), places=12)

class TestLineSearchToConstraints(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    # Helper to build a small, well-conditioned SPD problem
    def make_small_problem(self, B=1, n=3, dtype=torch.float64, device="cpu"):
        torch.manual_seed(0)
        A = torch.randn(B, n, n, dtype=dtype, device=device)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(
            n, dtype=dtype, device=device
        )  # SPD
        m = torch.randn(B, n, dtype=dtype, device=device)

        # Simple bounds that include zero
        L = torch.full((B, n), -1.0, dtype=dtype, device=device)
        U = torch.full((B, n), 2.0, dtype=dtype, device=device)

        # Budget weights and targets
        p = torch.randn(B, n, dtype=dtype, device=device)
        wl = torch.full((B,), 0.25, dtype=dtype, device=device)
        wh = torch.full((B,), 0.75, dtype=dtype, device=device)

        # Commission thresholds (not used here but required by __init__)
        c = torch.full((B,), 0.1, dtype=dtype, device=device)
        return Q, m, p, c, wl, wh, L, U

    def test_full_step_no_constraints(self):
        B, n = 1, 3
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=B, n=n)
        L = torch.full((B, n), -10.0)
        U = torch.full((B, n), 10.0)
        wl = torch.tensor([-100.0])
        wh = torch.tensor([100.0])

        x = torch.tensor([[0.0, 0.0, 0.0]])
        x_eq_proj = torch.tensor(
            [[1.0, -2.0, 3.0]]
        )  # well within bounds; zero-cross not triggered from x=0
        p = torch.tensor([[1.0, 2.0, 3.0]])
        active = torch.zeros_like(L, dtype=bool)
        budget_active = torch.zeros_like(wh, dtype=bool)

        x_new = ActiveSetQPFunc._line_search_to_constraints(
            x, x_eq_proj, p, wl, wh, L, U, active, budget_active
        )
        self.assertTrue(torch.allclose(x_new, x_eq_proj, atol=1e-12))

    def test_hits_upper_bound(self):
        B, n = 1, 2
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=B, n=n)
        L = torch.full((B, n), -10.0)
        U = torch.tensor([[1.0, 10.0]])
        wl = torch.tensor([-100.0])
        wh = torch.tensor([100.0])

        x = torch.tensor([[0.7, 0.0]])
        x_eq_proj = torch.tensor([[2.0, 5.0]])  # would exceed U[0,0]
        p = torch.tensor([[1.0, 1.0]])
        active = torch.zeros_like(L, dtype=bool)
        budget_active = torch.zeros_like(wh, dtype=bool)

        x_new = ActiveSetQPFunc._line_search_to_constraints(
            x, x_eq_proj, p, wl, wh, L, U, active, budget_active
        )
        # First coordinate should clamp to U=1.0; second moves proportionally
        self.assertAlmostEqual(float(x_new[0, 0]), 1.0, places=12)
        alpha = (1.0 - 0.7) / (2.0 - 0.7)  # expected alpha to hit U on coord 0
        self.assertAlmostEqual(
            float(x_new[0, 1]), float(0.0 + alpha * (5.0 - 0.0)), places=12
        )

    def test_hits_lower_bound(self):
        B, n = 1, 2
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=B, n=n)
        L = torch.tensor([[-1.0, -10.0]])
        U = torch.full((B, n), 10.0)
        wl = torch.tensor([-100.0])
        wh = torch.tensor([100.0])

        x = torch.tensor([[-0.7, 0.0]])
        x_eq_proj = torch.tensor([[-2.0, -5.0]])  # would go below L[0,0]
        p = torch.tensor([[1.0, 1.0]])
        active = torch.zeros_like(L, dtype=bool)
        budget_active = torch.zeros_like(wh, dtype=bool)

        x_new = ActiveSetQPFunc._line_search_to_constraints(
            x, x_eq_proj, p, wl, wh, L, U, active, budget_active
        )
        self.assertAlmostEqual(float(x_new[0, 0]), -1.0, places=12)
        alpha = (-1.0 - (-0.7)) / (-2.0 - (-0.7))
        self.assertAlmostEqual(
            float(x_new[0, 1]), float(0.0 + alpha * (-5.0 - 0.0)), places=12
        )

    def test_budget_low_plane_hit(self):
        B, n = 1, 2
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=B, n=n)
        L = torch.full((B, n), -10.0)
        U = torch.full((B, n), 10.0)
        wl = torch.tensor([1.0])
        wh = torch.tensor([10.0])

        x = torch.tensor([[2.0, 2.0]])  # p·x = 4
        x_eq_proj = torch.tensor([[0.0, 0.0]])  # p·delta = -4
        p = torch.tensor([[1.0, 1.0]])
        active = torch.zeros_like(L, dtype=bool)
        budget_active = torch.zeros_like(wh, dtype=bool)

        x_new = ActiveSetQPFunc._line_search_to_constraints(
            x, x_eq_proj, p, wl, wh, L, U, active, budget_active
        )
        p_dot = float((p * x_new).sum())
        self.assertAlmostEqual(p_dot, 1.0, places=12)  # hits wl
        # ensure no zero crossing before: alpha should be 0.75 < 1
        self.assertLess(
            float(torch.norm(x_new - x_eq_proj)),
            float(torch.norm(x - x_eq_proj)),
        )

    def test_budget_high_plane_hit(self):
        B, n = 1, 2
        Q, m, p, c, wl, wh, L, U = self.make_small_problem(B=B, n=n)
        L = torch.full((B, n), -10.0)
        U = torch.full((B, n), 10.0)
        wl = torch.tensor([-10.0])
        wh = torch.tensor([5.0])

        x = torch.tensor([[0.0, 0.0]])  # p·x = 0
        x_eq_proj = torch.tensor([[10.0, 10.0]])  # p·delta = 20
        p = torch.tensor([[1.0, 1.0]])
        active = torch.zeros_like(L, dtype=bool)
        budget_active = torch.zeros_like(wh, dtype=bool)

        x_new = ActiveSetQPFunc._line_search_to_constraints(
            x, x_eq_proj, p, wl, wh, L, U, active, budget_active
        )
        p_dot = float((p * x_new).sum())
        self.assertAlmostEqual(p_dot, 5.0, places=12)  # hits wh
        # alpha expected = (5 - 0)/20 = 0.25
        self.assertTrue(torch.allclose(x_new, x + 0.25 * (x_eq_proj - x), atol=1e-12))

    def test_batched_mixed(self):
        B, n = 2, 2
        Q, m, p, c, _, _, _, _ = self.make_small_problem(B=B, n=n)
        L = torch.tensor([[-10.0, -10.0], [-10.0, -10.0]])
        U = torch.tensor([[10.0, 10.0], [1.0, 10.0]])  # batch1 has tight U on coord0
        wl = torch.tensor([1.0, -10.0])
        wh = torch.tensor([5.0, 10.0])

        x = torch.tensor([[2.0, 2.0], [0.7, 0.0]])
        x_eq_proj = torch.tensor([[10.0, 10.0], [2.0, 5.0]])
        p = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        active = torch.zeros_like(L, dtype=bool)
        budget_active = torch.zeros_like(wh, dtype=bool)

        x_new = ActiveSetQPFunc._line_search_to_constraints(
            x, x_eq_proj, p, wl, wh, L, U, active, budget_active
        )

        # Batch 0: budget-high should stop at wh=5
        self.assertAlmostEqual(float((p[0] * x_new[0]).sum()), 5.0, places=12)

        # Batch 1: should stop at U[1,0]=1.0 on coord 0
        self.assertAlmostEqual(float(x_new[1, 0]), 1.0, places=12)


class TestComputeProjectedGradients(unittest.TestCase):
    def make_small_problem(self, B=2, n=3, dtype=torch.float64, device="cpu", seed=0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        A = torch.randn(B, n, n, dtype=dtype, device=device, generator=g)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(
            n, dtype=dtype, device=device
        )  # SPD
        H = 2 * Q
        m = torch.randn(B, n, dtype=dtype, device=device, generator=g)
        L = torch.full((B, n), -1.0, dtype=dtype, device=device)
        U = torch.full((B, n), 2.0, dtype=dtype, device=device)
        wl = torch.full((B,), 0.25, dtype=dtype, device=device)
        wh = torch.full((B,), 0.75, dtype=dtype, device=device)
        c = torch.full((B,), 0.1, dtype=dtype, device=device)
        p = torch.randn(B, n, dtype=dtype, device=device, generator=g)
        x = torch.randn(B, n, dtype=dtype, device=device, generator=g) * 0.2  # small
        return Q, H, m, L, U, wl, wh, c, p, x

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        # Instantiate your existing class (assumes it's already defined in the notebook)
        # from your_module import ActiveSetQPSolver  # if using modules
        Q, H, m, L, U, wl, wh, c, p, x = self.make_small_problem(B=2, n=3)
        self.Q, self.H, self.m = Q, H, m
        self.L, self.U, self.wl, self.wh, self.c = L, U, wl, wh, c
        self.p, self.x = p, x
        self.H = 2 * Q

        def compute_projected_gradients(x_new, p, active_comp, active_budget):
            return ActiveSetQPFunc._compute_projected_gradients(
                Q, m, x_new, p, active_comp, active_budget
            )

        self._compute_projected_gradients = compute_projected_gradients

    def test_no_budget_active_returns_raw_gradient(self):
        B, n = self.m.shape
        x_new = self.x.clone()
        p = self.p.clone()

        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.zeros(B, dtype=torch.bool)
        active_budg_high = torch.zeros(B, dtype=torch.bool)

        active = active_low | active_high | active_zero  # (B, n)
        budget_active = active_budg_low | active_budg_high  # (B,)
        g, g_along_p = self._compute_projected_gradients(
            x_new, p, active, budget_active
        )

        grad_expected = (self.H @ x_new.unsqueeze(-1)).squeeze(-1) + self.m
        self.assertTrue(torch.allclose(g, grad_expected, atol=1e-10, rtol=1e-10))

        # g_along_p should just be (p * grad).sum when nothing is pinned
        g_along_expected = (p * grad_expected).sum(dim=-1)
        self.assertTrue(
            torch.allclose(g_along_p, g_along_expected, atol=1e-10, rtol=1e-10)
        )

    def test_budget_active_projects_onto_tangent(self):
        B, n = self.m.shape
        x_new = self.x.clone()
        p = self.p.clone()

        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        # Activate budget on batch 0 only (low), none on batch 1
        active_budg_low = torch.tensor([True, False])
        active_budg_high = torch.tensor([False, False])

        active = active_low | active_high | active_zero  # (B, n)
        budget_active = active_budg_low | active_budg_high  # (B,)
        g, g_along_p = self._compute_projected_gradients(
            x_new, p, active, budget_active
        )

        grad = (self.H @ x_new.unsqueeze(-1)).squeeze(-1) + self.m
        denom = (p * p).sum(dim=-1).clamp_min(1e-30).unsqueeze(-1)
        proj = grad - ((p * grad).sum(dim=-1, keepdim=True) / denom) * p

        # Batch 0 projected, batch 1 unchanged
        self.assertTrue(torch.allclose(g[0], proj[0], atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(g[1], grad[1], atol=1e-10, rtol=1e-10))

        # Orthogonality on batch 0: p·g == 0
        self.assertAlmostEqual(float((p[0] * g[0]).sum()), 0.0, places=10)

    def test_pinned_coords_excluded_in_g_along_p(self):
        B, n = self.m.shape
        x_new = self.x.clone()
        p = self.p.clone()

        # Pin coordinate 1 across both batches (as zero constraints)
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_zero[:, 1] = True

        # Budget flags off (so g == grad)
        active_budg_low = torch.zeros(B, dtype=torch.bool)
        active_budg_high = torch.zeros(B, dtype=torch.bool)

        active = active_low | active_high | active_zero  # (B, n)
        budget_active = active_budg_low | active_budg_high  # (B,)
        g, g_along_p = self._compute_projected_gradients(
            x_new, p, active, budget_active
        )

        grad = (self.H @ x_new.unsqueeze(-1)).squeeze(-1) + self.m
        # p_eff masks out pinned coords
        p_eff = p.clone()
        p_eff[:, 1] = 0.0
        expected = (p_eff * grad).sum(dim=-1)

        self.assertTrue(torch.allclose(g_along_p, expected, atol=1e-10, rtol=1e-10))

    def test_batched_mixed(self):
        # Batch 0: budget active & no pins
        # Batch 1: no budget & pin coord 0
        B, n = self.m.shape
        x_new = self.x.clone()
        p = self.p.clone()

        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_zero[1, 0] = True

        active_budg_low = torch.tensor([True, False], dtype=torch.bool)
        active_budg_high = torch.tensor([False, False], dtype=torch.bool)

        active = active_low | active_high | active_zero  # (B, n)
        budget_active = active_budg_low | active_budg_high  # (B,)
        g, g_along_p = self._compute_projected_gradients(
            x_new, p, active, budget_active
        )

        grad = (self.H @ x_new.unsqueeze(-1)).squeeze(-1) + self.m

        # Batch 0: projected
        denom0 = (p[0] * p[0]).sum().clamp_min(1e-30)
        proj0 = grad[0] - ((p[0] * grad[0]).sum() / denom0) * p[0]
        self.assertTrue(torch.allclose(g[0], proj0, atol=1e-10, rtol=1e-10))
        self.assertAlmostEqual(float((p[0] * g[0]).sum()), 0.0, places=10)

        # Batch 1: raw gradient
        self.assertTrue(torch.allclose(g[1], grad[1], atol=1e-10, rtol=1e-10))

        # g_along_p with pin on coord 0 in batch 1
        p_eff1 = p[1].clone()
        p_eff1[0] = 0.0
        expected1 = (p_eff1 * grad[1]).sum()
        self.assertAlmostEqual(float(g_along_p[1]), float(expected1), places=10)

class TestUpdateActiveSets(unittest.TestCase):
    # Helper to make a tiny consistent problem (Q/m aren't used by _update_active_sets,
    # but we need them to construct the solver)
    def make_problem(self, B=2, n=3, dtype=torch.float64, device="cpu"):
        torch.manual_seed(0)
        A = torch.randn(B, n, n, dtype=dtype, device=device)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(
            n, dtype=dtype, device=device
        )  # SPD
        m = torch.randn(B, n, dtype=dtype, device=device)
        L = torch.full((B, n), -1.0, dtype=dtype, device=device)
        U = torch.full((B, n), 2.0, dtype=dtype, device=device)
        wl = torch.full((B,), 0.5, dtype=dtype, device=device)
        wh = torch.full((B,), 1.5, dtype=dtype, device=device)
        c = torch.full((B,), 0.3, dtype=dtype, device=device)
        p = torch.randn(B, n, dtype=dtype, device=device)
        return Q, m, L, U, wl, wh, c, p

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        Q, m, L, U, wl, wh, c, p = self.make_problem(B=2, n=3)
        self.Q, self.m, self.L, self.U = Q, m, L, U
        self.wl, self.wh, self.c, self.p = wl, wh, c, p

        def compute_active_sets(x_new, p, g, g_along_p, active_comp, active_budget):
            return ActiveSetQPFunc._compute_active_sets(
                x_new, p, g, g_along_p, Q, c, wl, wh, L, U, active_comp, active_budget
            )

        self._compute_active_sets = compute_active_sets
        self.eps = 1e-10

    def test_box_rules_lower_and_upper(self):
        B, n = self.L.shape
        eps = self.eps

        # x_new puts certain coords exactly on bounds
        x_new = torch.zeros(B, n)
        x_new[0] = torch.tensor(
            [self.L[0, 0], self.L[0, 1], self.U[0, 2]]
        )  # at low, at low, at high
        x_new[1] = torch.tensor(
            [0.1, 0.2, 0.3]
        )  # interior (won't be updated if hit=False)

        # Gradients: choose to exercise keep/drop logic
        # lower keep if g>=0; drop if g<0
        # upper keep if g<=0; drop if g>0
        g = torch.zeros(B, n)
        g[0] = torch.tensor(
            [+1.0, -1.0, -0.5]
        )  # keep lower (idx0), drop lower (idx1), keep upper (idx2)

        # g_along_p (not used for box logic)
        g_along_p = torch.zeros(B)

        # Initial masks (all False)
        active_low = torch.ones(B, n, dtype=torch.bool)
        active_high = torch.ones(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.ones(B, dtype=torch.bool)
        active_budg_high = torch.ones(B, dtype=torch.bool)

        # Only batch 0 "hit"; batch 1 should not be modified
        hit = torch.tensor([True, False], dtype=torch.bool)

        # Run
        (
            active_low,
            active_high,
            active_zero,
            active_budg_low,
            active_budg_high,
        ) = self._compute_active_sets(
            x_new,
            self.p,
            g,
            g_along_p,
            active_low | active_high,
            active_budg_low | active_budg_high,
        )

        # Batch 0 expectations
        self.assertTrue(bool(active_low[0, 0]))  # kept
        self.assertFalse(bool(active_low[0, 1]))  # dropped
        self.assertTrue(bool(active_high[0, 2]))  # kept

        # Batch 1 unchanged (was all False)
        self.assertFalse(active_low[1].any().item())
        self.assertFalse(active_high[1].any().item())

    def test_zero_constraint_keep_and_drop(self):
        B, n = self.L.shape
        eps = self.eps

        # Put coords at zero (within tol)
        x_new = torch.zeros(B, n)
        x_new[0] = torch.tensor([0.0, 0.0, 0.0])
        x_new[1] = torch.tensor(
            [0.5, -0.5, 0.0]
        )  # last coord at zero but batch 1 won't be updated

        # Choose g magnitudes relative to c to test keep/drop:
        # zero_keep iff |g_i| <= c + eps
        g = torch.zeros(B, n)
        g[0] = torch.tensor(
            [self.c[0].item() / 2, self.c[0].item() * 2, 0.0]  # keep  # drop
        )  # keep

        g_along_p = torch.zeros(B)

        active_low = torch.ones(B, n, dtype=torch.bool)
        active_high = torch.ones(B, n, dtype=torch.bool)
        active_zero = torch.ones(B, n, dtype=torch.bool)
        active_budg_low = torch.ones(B, dtype=torch.bool)
        active_budg_high = torch.ones(B, dtype=torch.bool)

        hit = torch.tensor([True, False], dtype=torch.bool)

        # Run
        (
            active_low,
            active_high,
            active_zero,
            active_budg_low,
            active_budg_high,
        ) = self._compute_active_sets(
            x_new,
            self.p,
            g,
            g_along_p,
            active_low | active_high,
            active_budg_low | active_budg_high,
        )

        # Batch 0: zero keep/drop per |g|
        self.assertTrue(bool(active_zero[0, 0]))  # kept
        self.assertFalse(bool(active_zero[0, 1]))  # dropped
        self.assertTrue(bool(active_zero[0, 2]))  # kept

    def test_budget_low_and_high_rules(self):
        B, n = self.L.shape
        eps = self.eps

        # Make x_new land on wl for batch 0 and on wh for batch 1
        p = self.p.clone()
        # Build x_new by scaling p to hit desired budgets exactly (simple construction)
        denom0 = (p[0] * p[0]).sum().clamp_min(1e-30)
        denom1 = (p[1] * p[1]).sum().clamp_min(1e-30)
        x_new = torch.zeros(B, n)
        x_new[0] = (self.wl[0] / denom0) * p[0]  # p·x = wl
        x_new[1] = (self.wh[1] / denom1) * p[1]  # p·x = wh

        # Set g_along_p to satisfy sign rules:
        #  - at wl keep if g·p >= 0
        #  - at wh keep if g·p <= 0
        g = torch.zeros(B, n)
        g_along_p = torch.zeros(B)
        g_along_p[0] = +0.1  # keep low
        g_along_p[1] = -0.2  # keep high

        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.zeros(B, dtype=torch.bool)
        active_budg_high = torch.zeros(B, dtype=torch.bool)

        hit = torch.tensor([True, True], dtype=torch.bool)

        # Run
        (
            active_low,
            active_high,
            active_zero,
            active_budg_low,
            active_budg_high,
        ) = self._compute_active_sets(
            x_new,
            self.p,
            g,
            g_along_p,
            active_low | active_high,
            active_budg_low | active_budg_high,
        )

        self.assertTrue(bool(active_budg_low[0]))  # p·x on wl and g·p >= 0
        self.assertTrue(bool(active_budg_high[1]))  # p·x on wh and g·p <= 0

    def test_no_update_when_not_hit(self):
        B, n = self.L.shape

        # Prepare masks with some True entries
        active_low = torch.zeros(B, n, dtype=torch.bool)
        active_high = torch.zeros(B, n, dtype=torch.bool)
        active_zero = torch.zeros(B, n, dtype=torch.bool)
        active_budg_low = torch.zeros(B, dtype=torch.bool)
        active_budg_high = torch.zeros(B, dtype=torch.bool)

        active_low[1, 0] = True
        active_high[1, 1] = True
        active_zero[1, 2] = True
        active_budg_low[1] = True

        x_new = torch.zeros(B, n)  # at zero/bounds won't matter; not updating batch 1
        g = torch.zeros(B, n)
        g_along_p = torch.zeros(B)
        p = self.p.clone()

        # Only batch 0 hit; batch 1 must remain unchanged
        hit = torch.tensor([True, False], dtype=torch.bool)

        # Run
        (
            active_low,
            active_high,
            active_zero,
            active_budg_low,
            active_budg_high,
        ) = self._compute_active_sets(
            x_new,
            self.p,
            g,
            g_along_p,
            active_low | active_high,
            active_budg_low | active_budg_high,
        )

        # Batch 1 masks unchanged
        self.assertTrue(torch.equal(active_low[1], active_low[1]))
        self.assertTrue(torch.equal(active_high[1], active_high[1]))
        self.assertTrue(torch.equal(active_zero[1], active_zero[1]))
        self.assertEqual(bool(active_budg_low[1]), bool(active_budg_low[1]))
        self.assertEqual(bool(active_budg_high[1]), bool(active_budg_high[1]))

class TestSolveMethod(unittest.TestCase):
    def spd_from_rand(self, B, n, dtype, device, seed=0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        A = torch.randn(B, n, n, dtype=dtype, device=device, generator=g)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(n, dtype=dtype, device=device)
        return Q

    def setup_random_problem(self, B, n, dtype, device, seed=42):
        Q = self.spd_from_rand(B, n, dtype, device, seed)
        H = Q * 2
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1000)
        m = torch.randn(B, n, dtype=dtype, device=device, generator=g)
        p = torch.abs(torch.randn(B, n, dtype=dtype, device=device, generator=g))
        return Q, H, m, p

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_unconstrained_matches_unconstrained_solution(self):
        B, n = 1, 4
        device, dtype = "cpu", torch.float64

        for i in range(100):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            L = torch.full((B, n), -1e6, dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)
            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.full((B,), 1e9, dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            x = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)

            # Expect solution of H x = -m with H = 2Q
            x_expected = torch.linalg.solve((2 * Q)[0], -m[0]).unsqueeze(0)
            self.assertTrue(torch.allclose(x, x_expected, atol=1e-8, rtol=1e-8))

    def test_pinned_lower_bound_respected_and_free_KKT(self):
        B, n = 1, 3
        device, dtype = "cpu", torch.float64

        # search for a problem with L0 < 0
        tests_count = 0
        for i in range(100, 200):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            x_uncon = torch.linalg.solve(H[0], -m[0])
            L0 = float(x_uncon[0].item() / 2)  # force bound to bind
            if L0 > 0:
                continue
            tests_count += 1

            L = torch.tensor([[L0, -1e6, -1e6]], dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)
            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.full((B,), 1e9, dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            x = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)

            # Check pinned coord
            self.assertAlmostEqual(float(x[0, 0]), L0, places=8)

            # Check reduced KKT on free coords
            F = torch.tensor([1, 2])
            Bidx = torch.tensor([0])
            H_FF = H[0][F][:, F]
            H_FB = H[0][F][:, Bidx]
            rhs = -(m[0][F] + (H_FB @ x[0, Bidx].unsqueeze(-1)).squeeze(-1))
            x_F_expected = torch.linalg.solve(H_FF, rhs)
            self.assertTrue(torch.allclose(x[0, F], x_F_expected, atol=1e-8, rtol=1e-8))
        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_budget_high_plane_solution(self):
        B, n = 1, 3
        device, dtype = "cpu", torch.float64

        # Set wide box
        L = torch.full((B, n), -1e6, dtype=dtype, device=device)
        U = torch.full((B, n), 1e6, dtype=dtype, device=device)

        # Choose wh so unconstrained violates wh: set wh = p·x_uncon - 1.0
        # search for a problem with wh>0
        tests_count = 0
        for i in range(200, 300):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            x_uncon = torch.linalg.solve(H[0], -m[0])
            budget_uncon = float((p[0] @ x_uncon).item())
            if budget_uncon < 0.0:
                continue
            tests_count += 1

            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.tensor([budget_uncon / 2.0], dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            x = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)

            # On high budget plane
            self.assertAlmostEqual(float((p[0] * x[0]).sum()), float(wh[0]), places=8)

            # Projected gradient orthogonal to p
            grad = (H @ x.unsqueeze(-1)).squeeze(-1) + m  # (B, n)
            proj = grad[0] - ((p[0] @ grad[0]) / (p[0] @ p[0])) * p[0]
            self.assertAlmostEqual(float((p[0] @ proj).item()), 0.0, places=7)

            # Matches closed-form equality-constrained solution
            # x = x0 + H^{-1} p * ((wh - p·x0) / (p·H^{-1}p))
            y = torch.linalg.solve(H[0], p[0])
            lam = (wh[0] - (p[0] @ x_uncon)) / (p[0] @ y)
            x_expected = x_uncon + lam * y
            self.assertTrue(torch.allclose(x[0], x_expected, atol=1e-8, rtol=1e-8))
        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    """
    def test_zero_commission_activation_hits_zero(self):
        B, n = 1, 2
        device, dtype = "cpu", torch.float64

        for i in range(300, 400):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            # Make m push x toward opposite signs so we cross zero
            m = torch.tensor([[0.5, -0.5]], dtype=dtype, device=device)
            p = torch.tensor(
                [[1.0, -1.0]], dtype=dtype, device=device
            )  # start along p has mixed signs

            # Wide box
            L = torch.full((B, n), -1e6, dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)

            # Budgets wide, commission large to keep zeros when hit
            wl = torch.tensor([-1e9], dtype=dtype, device=device)
            wh = torch.tensor([1e9], dtype=dtype, device=device)
            c = torch.tensor(
                [1e3], dtype=dtype, device=device
            )  # large commission → keep zero if hit

            x0 = find_start_point(wl, wh, L, U, p)
            x = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)

            # Expect at least one coordinate to be exactly zero due
            # to zero-commission activation
            # TODO it does not work when line search does not reach 0
            self.assertTrue((x.abs() <= 1e-6).any().item())

            # Wherever x_i == 0, |gradient_i| <= c (the keep rule)
            grad = (H @ x.unsqueeze(-1)).squeeze(-1) + m
            zero_mask = x.abs() <= 1e-9
            if zero_mask.any():
                self.assertTrue(
                    torch.all(torch.abs(grad[zero_mask]) <= c[0] + 1e-6).item()
                )
    """

    def test_batched_mixed_unconstrained_and_budget(self):
        B, n = 2, 3
        device, dtype = "cpu", torch.float64

        # Wide box
        L = torch.full((B, n), -1e6, dtype=dtype, device=device)
        U = torch.full((B, n), 1e6, dtype=dtype, device=device)

        tests_count = 0
        for i in range(400, 500):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            # For batch 0, make budget bind at wh; for batch 1 make budgets wide
            x0_uncon = torch.linalg.solve(H[0], -m[0])
            wh0 = (p[0] @ x0_uncon) / 2.0
            if wh0 <= 0:
                continue

            tests_count += 1
            wl = torch.tensor([-1e9, -1e9], dtype=dtype, device=device)
            wh = torch.tensor([wh0.item(), 1e9], dtype=dtype, device=device)
            c = torch.zeros(B, dtype=dtype, device=device)

            # solver = self.solver(Q, m, c, wl, wh, L, U, eps=1e-10)
            # x = solver(p, max_iter=30)
            x0 = find_start_point(wl, wh, L, U, p)
            x = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)

            self.assertAlmostEqual(
                float((p[0] * x[0]).sum()), float(wh[0]), places=8
            )  # !!!
            # Batch 1: equals unconstrained
            x1_expected = torch.linalg.solve(H[1], -m[1])
            self.assertTrue(torch.allclose(x[1], x1_expected, atol=1e-8, rtol=1e-8))
        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_example_1(self):
        Q = (
            torch.stack(
                [
                    torch.tensor([[1.0, -0.5], [-0.5, 2.0]]),
                    torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
                ],
                dim=0,
            )
            * 2
        )
        m = -torch.stack([torch.tensor([8.0, 2.0]), torch.tensor([1.0, 0.0])], dim=0)
        p = torch.stack([torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])], dim=0)
        c = torch.tensor([0.0, 0.0])
        wl = torch.tensor([-1.5, -1.0])
        wh = torch.tensor([1.5, 1.0])
        U = torch.stack([torch.tensor([10.0, 10.0]), torch.tensor([0.6, 0.7])], dim=0)
        L = torch.stack([torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])], dim=0)
        x0 = find_start_point(wl, wh, L, U, p)
        real = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)
        exp = torch.tensor([[1.3125, 0.1875], [0.1250, 0.0000]])
        self.assertTrue(torch.allclose(real, exp))

    def test_example_2(self):
        Q1 = torch.tensor(
            [
                [0.07371012, 0.03541838, 0.01831166],
                [0.03541838, 0.05724201, 0.02421184],
                [0.01831166, 0.02421184, 0.01525362],
            ]
        ).unsqueeze(0)

        m1 = torch.tensor([-50.15707481, -49.95627699, -50.2048171]).unsqueeze(0)
        p1 = torch.tensor([50.0175, 49.9160, 49.9198]).unsqueeze(0)
        c = torch.zeros(1)
        wh1 = torch.tensor(164838.6090).unsqueeze(0)
        wl1 = torch.tensor(-164838.6090).unsqueeze(0)
        L1 = torch.tensor([-408.23219245, -409.06294104, -409.03185001]).unsqueeze(0)
        U1 = torch.tensor([2060.69218141, 2064.8856702, 2064.72872755]).unsqueeze(0)
        x0 = find_start_point(wl1, wh1, L1, U1, p1)
        real = solve_QP_problem(Q1, m1, c, wl1, wh1, L1, U1, p1, x0)
        exp = torch.tensor([[23.8539, -409.0629, 2064.7287]])
        self.assertTrue(torch.allclose(real, exp))

class TestBackwardMethod(unittest.TestCase):
    def spd_from_rand(self, B, n, dtype, device, seed=0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        A = torch.randn(B, n, n, dtype=dtype, device=device, generator=g)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(
            n, dtype=dtype, device=device
        )
        return Q

    def setup_random_problem(self, B, n, dtype, device, seed=42):
        Q = self.spd_from_rand(B, n, dtype, device, seed)
        H = Q * 2
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1000)
        m = torch.randn(B, n, dtype=dtype, device=device, generator=g)
        p = torch.abs(
            torch.randn(B, n, dtype=dtype, device=device, generator=g)
        )
        return Q, H, m, p

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def check_gradient(self, Q, m, c, wl, wh, L, U, p, x0):
        p = p.detach()
        p.requires_grad_(True)

        x = solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0)
        x.sum().backward()

        p1 = p.clone().detach()
        p1.requires_grad_(True)

        active_comp = (x == L) | (x == U)
        active_values = torch.where(
            active_comp, torch.where(x == L, L, U), 0.0
        )
        active_budget = (
            torch.abs(bdot(x, p) - wl) < 1e-9
        ) | (torch.abs(bdot(x, p) - wh) < 1e-9)
        budget_w = torch.where(
            active_budget,
            torch.where(
                torch.abs(bdot(x, p) - wl) < 1e-9,
                wl,
                wh,
            ),
            0.0,
        )

        x1 = ActiveSetQPFunc._solve_under_active(
            Q, m, p, active_comp, active_values, active_budget, budget_w
        )
        x1.sum().backward()

        self.assertTrue(torch.allclose(p, p1, atol=1e-8, rtol=1e-8))

    def test_unconstrained_matches_unconstrained_solution(self):
        B, n = 1, 4
        device, dtype = "cpu", torch.float64

        for i in range(100):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            L = torch.full((B, n), -1e6, dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)
            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.full((B,), 1e9, dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_gradient(Q, m, c, wl, wh, L, U, p, x0)

    def test_pinned_lower_bound_respected_and_free_KKT(self):
        B, n = 1, 3
        device, dtype = "cpu", torch.float64

        # search for a problem with L0 < 0
        tests_count = 0
        for i in range(100, 200):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            x_uncon = torch.linalg.solve(H[0], -m[0])
            L0 = float(x_uncon[0].item() / 2)  # force bound to bind
            if L0 > 0:
                continue
            tests_count += 1

            L = torch.tensor([[L0, -1e6, -1e6]], dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)
            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.full((B,), 1e9, dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_gradient(Q, m, c, wl, wh, L, U, p, x0)

        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_budget_high_plane_solution(self):
        B, n = 1, 3
        device, dtype = "cpu", torch.float64

        # Set wide box
        L = torch.full((B, n), -1e6, dtype=dtype, device=device)
        U = torch.full((B, n), 1e6, dtype=dtype, device=device)

        # Choose wh so unconstrained violates wh: set wh = p·x_uncon - 1.0
        # search for a problem with wh>0
        tests_count = 0
        for i in range(200, 300):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            x_uncon = torch.linalg.solve(H[0], -m[0])
            budget_uncon = float((p[0] @ x_uncon).item())
            if budget_uncon < 0.0:
                continue
            tests_count += 1

            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.tensor([budget_uncon / 2.0], dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_gradient(Q, m, c, wl, wh, L, U, p, x0)

        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_zero_commission_activation_hits_zero(self):
        B, n = 1, 2
        device, dtype = "cpu", torch.float64

        for i in range(300, 400):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            # Make m push x toward opposite signs so we cross zero
            m = torch.tensor([[0.5, -0.5]], dtype=dtype, device=device)
            p = torch.tensor(
                [[1.0, -1.0]], dtype=dtype, device=device
            )  # start along p has mixed signs

            # Wide box
            L = torch.full((B, n), -1e6, dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)

            # Budgets wide, commission large to keep zeros when hit
            wl = torch.tensor([-1e9], dtype=dtype, device=device)
            wh = torch.tensor([1e9], dtype=dtype, device=device)
            c = torch.tensor(
                [1e3], dtype=dtype, device=device
            )  # large commission → keep zero if hit

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_gradient(Q, m, c, wl, wh, L, U, p, x0)

    def test_batched_mixed_unconstrained_and_budget(self):
        B, n = 2, 3
        device, dtype = "cpu", torch.float64

        # Wide box
        L = torch.full((B, n), -1e6, dtype=dtype, device=device)
        U = torch.full((B, n), 1e6, dtype=dtype, device=device)

        tests_count = 0
        for i in range(400, 500):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            # For batch 0, make budget bind at wh; for batch 1 make budgets wide
            x0_uncon = torch.linalg.solve(H[0], -m[0])
            wh0 = (p[0] @ x0_uncon) / 2.0
            if wh0 <= 0:
                continue

            tests_count += 1
            wl = torch.tensor([-1e9, -1e9], dtype=dtype, device=device)
            wh = torch.tensor([wh0.item(), 1e9], dtype=dtype, device=device)
            c = torch.zeros(B, dtype=dtype, device=device)

            # solver = self.solver(Q, m, c, wl, wh, L, U, eps=1e-10)
            # x = solver(p, max_iter=30)
            x0 = find_start_point(wl, wh, L, U, p)
            self.check_gradient(Q, m, c, wl, wh, L, U, p, x0)

        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_example_1(self):
        Q = (
            torch.stack(
                [
                    torch.tensor([[1.0, -0.5], [-0.5, 2.0]]),
                    torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
                ],
                dim=0,
            )
            * 2
        )
        m = -torch.stack(
            [torch.tensor([8.0, 2.0]), torch.tensor([1.0, 0.0])], dim=0
        )
        p = torch.stack(
            [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])], dim=0
        )
        c = torch.tensor([0.0, 0.0])
        wl = torch.tensor([-1.5, -1.0])
        wh = torch.tensor([1.5, 1.0])
        U = torch.stack(
            [torch.tensor([10.0, 10.0]), torch.tensor([0.6, 0.7])], dim=0
        )
        L = torch.stack(
            [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])], dim=0
        )
        x0 = find_start_point(wl, wh, L, U, p)
        self.check_gradient(Q, m, c, wl, wh, L, U, p, x0)

    def test_example_2(self):
        Q1 = torch.tensor(
            [
                [0.07371012, 0.03541838, 0.01831166],
                [0.03541838, 0.05724201, 0.02421184],
                [0.01831166, 0.02421184, 0.01525362],
            ]
        ).unsqueeze(0)

        m1 = torch.tensor([-50.15707481, -49.95627699, -50.2048171]).unsqueeze(
            0
        )
        p1 = torch.tensor([50.0175, 49.9160, 49.9198]).unsqueeze(0)
        c = torch.zeros(1)
        wh1 = torch.tensor(164838.6090).unsqueeze(0)
        wl1 = torch.tensor(-164838.6090).unsqueeze(0)
        L1 = torch.tensor(
            [-408.23219245, -409.06294104, -409.03185001]
        ).unsqueeze(0)
        U1 = torch.tensor(
            [2060.69218141, 2064.8856702, 2064.72872755]
        ).unsqueeze(0)
        x0 = find_start_point(wl1, wh1, L1, U1, p1)
        self.check_gradient(Q1, m1, c, wl1, wh1, L1, U1, p1, x0)

class TestJVPMethod(unittest.TestCase):
    def spd_from_rand(self, B, n, dtype, device, seed=0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        A = torch.randn(B, n, n, dtype=dtype, device=device, generator=g)
        Q = A.transpose(-2, -1) @ A + 0.5 * torch.eye(n, dtype=dtype, device=device)
        return Q

    def setup_random_problem(self, B, n, dtype, device, seed=42):
        Q = self.spd_from_rand(B, n, dtype, device, seed)
        H = Q * 2
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1000)
        m = torch.randn(B, n, dtype=dtype, device=device, generator=g)
        p = torch.abs(torch.randn(B, n, dtype=dtype, device=device, generator=g))
        return Q, H, m, p

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def check_jvp(self, Q, m, c, wl, wh, L, U, p, x0):
        torch.manual_seed(0)
        rep = p.shape[0]

        f = lambda p: solve_QP_problem2(
            Q, m, c, wl, wh, L, U, p.repeat(rep, 1), x0
        ).sum(dim=0)

        x = p[0].clone()
        x.requires_grad_(True)

        # random test vectors
        v = torch.rand_like(x)  # tangent for inputs
        u = torch.rand_like(x)  # cotangent for outputs

        # --- Jv via forward-mode (JVP) ---
        with dual_level():
            x_dual = make_dual(x, v)
            y_dual = f(x_dual)
            y, Jv = unpack_dual(y_dual)

        # --- J^T u via reverse-mode (VJP) ---
        (JT_u,) = grad(y, (x,), grad_outputs=u, retain_graph=False, create_graph=False)

        # Inner-product check
        lhs = (u * Jv).sum()  # <u, Jv>
        rhs = (JT_u * v).sum()  # <J^T u, v>
        # print(lhs, rhs)
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-8, rtol=1e-8))

    def test_unconstrained_matches_unconstrained_solution(self):
        B, n = 1, 4
        device, dtype = "cpu", torch.float64

        for i in range(100):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            L = torch.full((B, n), -1e6, dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)
            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.full((B,), 1e9, dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_jvp(Q, m, c, wl, wh, L, U, p, x0)

    def test_pinned_lower_bound_respected_and_free_KKT(self):
        B, n = 1, 3
        device, dtype = "cpu", torch.float64

        # search for a problem with L0 < 0
        tests_count = 0
        for i in range(100, 200):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            x_uncon = torch.linalg.solve(H[0], -m[0])
            L0 = float(x_uncon[0].item() / 2)  # force bound to bind
            if L0 > 0:
                continue
            tests_count += 1

            L = torch.tensor([[L0, -1e6, -1e6]], dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)
            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.full((B,), 1e9, dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_jvp(Q, m, c, wl, wh, L, U, p, x0)

        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_budget_high_plane_solution(self):
        B, n = 1, 3
        device, dtype = "cpu", torch.float64

        # Set wide box
        L = torch.full((B, n), -1e6, dtype=dtype, device=device)
        U = torch.full((B, n), 1e6, dtype=dtype, device=device)

        # Choose wh so unconstrained violates wh: set wh = p·x_uncon - 1.0
        # search for a problem with wh>0
        tests_count = 0
        for i in range(200, 300):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            x_uncon = torch.linalg.solve(H[0], -m[0])
            budget_uncon = float((p[0] @ x_uncon).item())
            if budget_uncon < 0.0:
                continue
            tests_count += 1

            wl = torch.full((B,), -1e9, dtype=dtype, device=device)
            wh = torch.tensor([budget_uncon / 2.0], dtype=dtype, device=device)
            c = torch.full((B,), 0.0, dtype=dtype, device=device)

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_jvp(Q, m, c, wl, wh, L, U, p, x0)

        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_zero_commission_activation_hits_zero(self):
        B, n = 1, 2
        device, dtype = "cpu", torch.float64

        for i in range(300, 400):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            # Make m push x toward opposite signs so we cross zero
            m = torch.tensor([[0.5, -0.5]], dtype=dtype, device=device)
            p = torch.tensor(
                [[1.0, -1.0]], dtype=dtype, device=device
            )  # start along p has mixed signs

            # Wide box
            L = torch.full((B, n), -1e6, dtype=dtype, device=device)
            U = torch.full((B, n), 1e6, dtype=dtype, device=device)

            # Budgets wide, commission large to keep zeros when hit
            wl = torch.tensor([-1e9], dtype=dtype, device=device)
            wh = torch.tensor([1e9], dtype=dtype, device=device)
            c = torch.tensor(
                [1e3], dtype=dtype, device=device
            )  # large commission → keep zero if hit

            x0 = find_start_point(wl, wh, L, U, p)
            self.check_jvp(Q, m, c, wl, wh, L, U, p, x0)

    def test_batched_mixed_unconstrained_and_budget(self):
        B, n = 2, 3
        device, dtype = "cpu", torch.float64

        # Wide box
        L = torch.full((B, n), -1e6, dtype=dtype, device=device)
        U = torch.full((B, n), 1e6, dtype=dtype, device=device)

        tests_count = 0
        for i in range(400, 500):
            Q, H, m, p = self.setup_random_problem(B, n, dtype, device, seed=i)
            # For batch 0, make budget bind at wh; for batch 1 make budgets wide
            x0_uncon = torch.linalg.solve(H[0], -m[0])
            wh0 = (p[0] @ x0_uncon) / 2.0
            if wh0 <= 0:
                continue

            tests_count += 1
            wl = torch.tensor([-1e9, -1e9], dtype=dtype, device=device)
            wh = torch.tensor([wh0.item(), 1e9], dtype=dtype, device=device)
            c = torch.zeros(B, dtype=dtype, device=device)

            # solver = self.solver(Q, m, c, wl, wh, L, U, eps=1e-10)
            # x = solver(p, max_iter=30)
            x0 = find_start_point(wl, wh, L, U, p)
            self.check_jvp(Q, m, c, wl, wh, L, U, p, x0)

        # check that at least one test performed
        self.assertGreater(tests_count, 0)

    def test_example_1(self):
        Q = (
            torch.stack(
                [
                    torch.tensor([[1.0, -0.5], [-0.5, 2.0]]),
                    torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
                ],
                dim=0,
            )
            * 2
        )
        m = -torch.stack([torch.tensor([8.0, 2.0]), torch.tensor([1.0, 0.0])], dim=0)
        p = torch.stack([torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])], dim=0)
        c = torch.tensor([0.0, 0.0])
        wl = torch.tensor([-1.5, -1.0])
        wh = torch.tensor([1.5, 1.0])
        U = torch.stack([torch.tensor([10.0, 10.0]), torch.tensor([0.6, 0.7])], dim=0)
        L = torch.stack([torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])], dim=0)
        x0 = find_start_point(wl, wh, L, U, p)
        self.check_jvp(Q, m, c, wl, wh, L, U, p, x0)

    def test_example_2(self):
        Q1 = torch.tensor(
            [
                [0.07371012, 0.03541838, 0.01831166],
                [0.03541838, 0.05724201, 0.02421184],
                [0.01831166, 0.02421184, 0.01525362],
            ]
        ).unsqueeze(0)

        m1 = torch.tensor([-50.15707481, -49.95627699, -50.2048171]).unsqueeze(0)
        p1 = torch.tensor([50.0175, 49.9160, 49.9198]).unsqueeze(0)
        c = torch.zeros(1)
        wh1 = torch.tensor(164838.6090).unsqueeze(0)
        wl1 = torch.tensor(-164838.6090).unsqueeze(0)
        L1 = torch.tensor([-408.23219245, -409.06294104, -409.03185001]).unsqueeze(0)
        U1 = torch.tensor([2060.69218141, 2064.8856702, 2064.72872755]).unsqueeze(0)
        x0 = find_start_point(wl1, wh1, L1, U1, p1)
        self.check_jvp(Q1, m1, c, wl1, wh1, L1, U1, p1, x0)


if __name__ == "__main__":
    unittest.main()
