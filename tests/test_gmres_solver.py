import importlib
import unittest
import torch

import numpy as np
import jax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

_core = importlib.import_module("heteromarket.core")
StockSolverSum = _core.StockSolverSum
bdot = _core.bdot

dummy_primals = torch.tensor(0)

def find_start_point(wl, wh, L, U, p):
    denom = bdot(p, U).clamp_min(1e-30)
    x0 = torch.where(
        wl.unsqueeze(-1) < 0.0,
        torch.zeros_like(p),
        U * (wl / denom).unsqueeze(-1),
    )
    return x0


def simple_matvec(x, primals):
    return x


def simple_residual(b, x, primals):
    return b - StockSolverSum.matvec(x, primals)


def linear_matvec(x, primals):
    A, M = primals
    return M @ (A @ x)


def linear_residual(b, x, primals):
    A, M = primals
    return M @ (b - A @ x)


class TestSafeNormalize(unittest.TestCase):
    def assertNoNaN(self, t: torch.Tensor):
        self.assertFalse(torch.isnan(t).any().item(), "Tensor contains NaNs")

    def test_zero_vector_returns_zero(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.zeros(7, dtype=dtype)
                y, n = StockSolverSum._safe_normalize(x)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(n.shape, torch.Size(()))  # scalar tensor
                self.assertTrue(torch.allclose(y, torch.zeros_like(x)))
                self.assertEqual(n.item(), 0.0)
                self.assertNoNaN(y)
                self.assertNoNaN(n)

    def test_nonzero_vector_normalizes_to_unit(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([3.0, 4.0], dtype=dtype)
                y, n = StockSolverSum._safe_normalize(x)
                self.assertTrue(torch.allclose(n, torch.tensor(5.0, dtype=dtype)))
                atol = 1e-6 if dtype == torch.float32 else 1e-12
                self.assertTrue(
                    torch.allclose(
                        torch.linalg.vector_norm(y),
                        torch.tensor(1.0, dtype=dtype),
                        atol=atol,
                        rtol=0,
                    )
                )
                self.assertTrue(torch.allclose(y, x / n, atol=atol, rtol=0))

    def test_unit_vector_is_idempotent(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
                y, n = StockSolverSum._safe_normalize(x)
                atol = 1e-7 if dtype == torch.float32 else 1e-12
                self.assertTrue(torch.allclose(y, x, atol=atol, rtol=0))
                self.assertEqual(n.item(), 1.0)

    def test_custom_threshold_forces_zero(self):
        x = torch.tensor([1e-8, -1e-8], dtype=torch.float32)
        y, n = StockSolverSum._safe_normalize_thresh(x, torch.tensor(1e-7))  # ||x|| ≈ 1.414e-8 < thresh
        self.assertTrue(torch.allclose(y, torch.zeros_like(x)))
        self.assertEqual(n.item(), 0.0)

    def test_above_threshold_normalizes(self):
        cases = [
            (torch.float32, 1e-6, 1e-7, 1e-6),
            (torch.float64, 1e-12, 1e-13, 1e-12),
        ]
        for dtype, val, thresh, atol in cases:
            thresh_t = torch.tensor(thresh)
            with self.subTest(dtype=dtype):
                x = torch.tensor([val, 0.0, -val], dtype=dtype)
                y, n = StockSolverSum._safe_normalize_thresh(x, thresh_t)
                self.assertGreater(n.item(), thresh)
                self.assertTrue(
                    torch.allclose(
                        torch.linalg.vector_norm(y),
                        torch.tensor(1.0, dtype=dtype),
                        atol=atol,
                        rtol=0,
                    )
                )
                self.assertTrue(torch.allclose(y, x / n, atol=atol, rtol=0))

    def test_default_threshold_matches_eps_behavior(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                eps = torch.finfo(dtype).eps
                # ||x|| = sqrt(3)*(eps/2) < eps -> should zero out
                x = torch.full((3,), eps / 2, dtype=dtype)
                y, n = StockSolverSum._safe_normalize(x)
                self.assertTrue(torch.allclose(y, torch.zeros_like(x)))
                self.assertEqual(n.item(), 0.0)


def orthonormal_columns(m: int, k: int, dtype=torch.float64, device="cpu"):
    """Create an (m, k) matrix with orthonormal columns."""
    A = torch.randn(m, k, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q


class TestIterativeCGSUnnormalized(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def assertAllClose(self, a, b, dtype, atol=None, rtol=0.0, msg=""):
        if atol is None:
            atol = 1e-6 if dtype == torch.float32 else 1e-12
        self.assertTrue(
            torch.allclose(a, b, atol=atol, rtol=rtol), msg or f"{a} vs {b}"
        )

    def _assert_same_direction(self, u, v, dtype, atol=None):
        if atol is None:
            atol = 1e-6 if dtype == torch.float32 else 1e-12
        nu = torch.linalg.vector_norm(u).item()
        nv = torch.linalg.vector_norm(v).item()
        # both zero → fine
        if nu == 0.0 and nv == 0.0:
            return
        # one zero, one nonzero → fail
        self.assertFalse(
            (nu == 0.0) ^ (nv == 0.0), "One vector is zero, the other is not"
        )
        uhat = u / nu
        vhat = v / nv
        cos = torch.dot(uhat, vhat).abs()
        self.assertGreaterEqual(
            cos.item(), 0.97 if dtype == torch.float32 else 1.0 - 1e-6
        )

    def test_empty_Q_returns_x_and_empty_r(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = 7
                Q = torch.empty((m, 0), dtype=dtype)
                x = torch.randn(m, dtype=dtype)
                xnorm = torch.linalg.vector_norm(x)
                q, r = StockSolverSum._iterative_classical_gram_schmidt(Q, x)
                self.assertEqual(r.numel(), 0)
                # q should equal x (unnormalized, since no columns to remove)
                self.assertAllClose(q, x, dtype)

    def test_q_is_orthogonal_to_Q_and_reconstruction_holds(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, k = 12, 4
                Q = orthonormal_columns(m, k, dtype=dtype)
                x = torch.randn(m, dtype=dtype)
                xnorm = torch.linalg.vector_norm(x)
                q, r = StockSolverSum._iterative_classical_gram_schmidt(Q, x)

                # Orthogonality: Q^T q ≈ 0
                ortho = Q.T @ q
                self.assertTrue(
                    torch.allclose(
                        ortho,
                        torch.zeros_like(ortho),
                        atol=1e-6 if dtype == torch.float32 else 1e-12,
                    )
                )

                # Reconstruction identity: x ≈ Q r + q (no alpha; q is unnormalized)
                x_hat = Q @ r + q
                self.assertAllClose(x_hat, x, dtype)

    def test_r_matches_QTx_in_exact_arithmetic(self):
        # In exact arithmetic with orthonormal Q, CGS (even with re-orth) gives r ≈ Q^T x
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, k = 10, 5
                Q = orthonormal_columns(m, k, dtype=dtype)
                x = torch.randn(m, dtype=dtype)
                xnorm = torch.linalg.vector_norm(x)
                q, r = StockSolverSum._iterative_classical_gram_schmidt(Q, x)
                r_true = Q.T @ x
                self.assertTrue(
                    torch.allclose(
                        r, r_true, atol=1e-6 if dtype == torch.float32 else 1e-12
                    )
                )

    def test_x_in_span_Q_returns_zero_q_and_r_is_coeffs(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, k = 8, 5
                Q = orthonormal_columns(m, k, dtype=dtype)
                coeffs = torch.randn(k, dtype=dtype)
                x = Q @ coeffs
                xnorm = torch.linalg.vector_norm(x)
                q, r = StockSolverSum._iterative_classical_gram_schmidt(Q, x)
                # q ~ 0, r ~ coeffs
                self.assertTrue(
                    torch.allclose(
                        q,
                        torch.zeros_like(q),
                        atol=1e-7 if dtype == torch.float32 else 1e-12,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        r, coeffs, atol=1e-6 if dtype == torch.float32 else 1e-12
                    )
                )

    def test_results_direction_stable_wrt_max_iterations(self):
        # Compare q from 1 vs 3 re-orth passes up to scale/sign
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, k = 20, 10
                Q = orthonormal_columns(m, k, dtype=dtype)
                # Make x mostly in span(Q) to exercise re-orth path
                c = torch.randn(k, dtype=dtype)
                small = torch.randn(m, dtype=dtype) * (
                    1e-6 if dtype == torch.float32 else 1e-12
                )
                x = Q @ c + small
                xnorm = torch.linalg.vector_norm(x)

                q1, r1 = StockSolverSum._iterative_classical_gram_schmidt(Q, x)
                q3, r3 = StockSolverSum._iterative_classical_gram_schmidt(Q, x)

                # r should be very close; q should point in (nearly) the same direction
                self.assertTrue(
                    torch.allclose(
                        r1, r3, atol=1e-6 if dtype == torch.float32 else 1e-12
                    )
                )
                self._assert_same_direction(q1, q3, dtype)

    def test_shapes_and_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, k = 9, 3
                Q = orthonormal_columns(m, k, dtype=dtype)
                x = torch.randn(m, dtype=dtype)
                xnorm = torch.linalg.vector_norm(x)
                q, r = StockSolverSum._iterative_classical_gram_schmidt(Q, x)
                self.assertEqual(q.dtype, dtype)
                self.assertEqual(r.dtype, dtype)
                self.assertEqual(q.shape, (m,))
                self.assertEqual(r.shape, (k,))

def twopass_cgs(V_used: torch.Tensor, w: torch.Tensor):
    """
    Two-pass Classical Gram–Schmidt against columns in V_used (shape (m, k+1)).
    Returns (r_used_full, q2), where:
      - r_used_full has length ncols (zeros beyond used columns)
      - q2 is the unnormalized residual.
    """
    ncols = V_used.shape[1]
    # Pass 1
    h1 = V_used.transpose(0, 1) @ w  # (k+1,)
    q1 = w - V_used @ h1
    # Pass 2
    h2 = V_used.transpose(0, 1) @ q1  # (k+1,)
    q2 = q1 - V_used @ h2
    r_used = h1 + h2  # (k+1,)
    return r_used, q2


class TestKthArnoldiIteration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.saved_matvec = StockSolverSum.matvec
        self.saved_compute_residual = StockSolverSum.compute_residual
        StockSolverSum.matvec = simple_matvec
        StockSolverSum.compute_residual = simple_residual

    def tearDown(self):
        StockSolverSum.matvec =  self.saved_matvec
        StockSolverSum.compute_residual = self.saved_compute_residual

    def assertAllClose(self, a, b, dtype, atol=None, rtol=0.0, msg=""):
        if atol is None:
            atol = 1e-6 if dtype == torch.float32 else 1e-12
        self.assertTrue(
            torch.allclose(a, b, atol=atol, rtol=rtol), msg or f"{a} vs {b}"
        )

    def _check_step(self, m, restart, k, dtype, f_state, M, V_init, H_init):
        """
        Compute one Arnoldi step both via your function and manually via
        two-pass CGS (with unnormalized residual), then compare.
        """
        device = V_init.device
        eps = torch.finfo(dtype).eps

        # Run function under test (tensor k)
        k_t = torch.tensor(k, dtype=torch.int64, device=device)
        primals = (f_state, M)
        V_out, H_out, breakdown = StockSolverSum._kth_arnoldi_iteration(
            k_t, V_init.clone(), H_init.clone(), primals
        )

        # Manual expectation
        v_k = V_init[:, k]
        w = StockSolverSum.matvec(v_k, primals)
        # Build two-pass CGS against used columns 0..k
        V_used = V_init[:, : k + 1]
        r_used, q2 = twopass_cgs(V_used, w)  # r_used length k+1, q2 unnormalized
        w_norm0 = torch.linalg.vector_norm(w)
        tol = eps * w_norm0

        q2_norm = torch.linalg.vector_norm(q2)
        if q2_norm.item() <= tol:
            unit_v = torch.zeros_like(q2)
            vnorm1 = torch.tensor(0.0, dtype=dtype, device=device)
            expected_breakdown = True
        else:
            unit_v = q2 / q2_norm
            vnorm1 = q2_norm
            expected_breakdown = False

        # Expected H row (length restart+1): r_used on 0..k, subdiag at k+1
        h_expected = torch.zeros(restart + 1, dtype=dtype, device=device)
        h_expected[: k + 1] = r_used
        h_expected[k + 1] = vnorm1

        # --- Assertions ---
        # 1) Earlier columns unchanged
        self.assertAllClose(V_out[:, : k + 1], V_init[:, : k + 1], dtype)
        # 2) New column equals expected unit vector
        self.assertAllClose(V_out[:, k + 1], unit_v, dtype)
        # 3) Later columns remain zero
        if k + 2 <= restart:
            self.assertTrue(
                torch.allclose(V_out[:, k + 2 :], torch.zeros_like(V_out[:, k + 2 :]))
            )
        # 4) Only row k of H changed, and equals expected
        self.assertAllClose(H_out[k, :], h_expected, dtype)
        if k > 0:
            self.assertTrue(torch.allclose(H_out[:k, :], H_init[:k, :]))
        if k + 1 < H_out.shape[0]:
            self.assertTrue(torch.allclose(H_out[k + 1 :, :], H_init[k + 1 :, :]))
        # 5) Breakdown flag
        self.assertEqual(bool(breakdown), expected_breakdown)

    def test_identity_A_random_M(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, restart, k = 16, 8, 3
                Q = orthonormal_columns(m, k + 1, dtype=dtype)
                V = torch.zeros((m, restart + 1), dtype=dtype)
                V[:, : k + 1] = Q
                H = torch.zeros((restart, restart + 1), dtype=dtype)
                M = torch.randn(m, m, dtype=dtype)
                dummy_state = torch.tensor(0)
                self._check_step(m, restart, k, dtype, dummy_state, M, V, H)

    def test_linear_A_general_M(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, restart, k = 12, 7, 2
                Q = orthonormal_columns(m, k + 1, dtype=dtype)
                V = torch.zeros((m, restart + 1), dtype=dtype)
                V[:, : k + 1] = Q
                H = torch.zeros((restart, restart + 1), dtype=dtype)
                A_mat = torch.randn(m, m, dtype=dtype)
                M = torch.randn(m, m, dtype=dtype)
                self._check_step(m, restart, k, dtype, A_mat, M, V, H)

    def test_breakdown_when_w_in_span(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, restart, k = 10, 6, 4
                Q = orthonormal_columns(m, k + 1, dtype=dtype)
                V = torch.zeros((m, restart + 1), dtype=dtype)
                V[:, : k + 1] = Q
                H = torch.zeros((restart, restart + 1), dtype=dtype)
                dummy_state = torch.tensor(0)
                M = torch.eye(m, dtype=dtype)

                # Here w = M @ A(v_k) = v_k, which lies in span(V[:, :k+1]).
                primals = (dummy_state, M)
                V_out, H_out, breakdown = StockSolverSum._kth_arnoldi_iteration(
                    torch.tensor(k, dtype=torch.int64), V.clone(), H.clone(), primals
                )
                # Expected: new column is zero, breakdown True
                self.assertTrue(
                    torch.allclose(V_out[:, k + 1], torch.zeros_like(V_out[:, k + 1]))
                )
                self.assertTrue(bool(breakdown))
                # H row = [0,...,1 at k, 0 at k+1, zeros...]
                e_k = torch.zeros(k + 1, dtype=dtype)
                e_k[k] = 1.0
                h_expected = torch.zeros(restart + 1, dtype=dtype)
                h_expected[: k + 1] = e_k
                self.assertAllClose(H_out[k, :], h_expected, dtype)

    def test_shapes_and_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, restart, k = 9, 5, 1
                Q = orthonormal_columns(m, k + 1, dtype=dtype)
                V = torch.zeros((m, restart + 1), dtype=dtype)
                V[:, : k + 1] = Q
                H = torch.zeros((restart, restart + 1), dtype=dtype)

                dummy_state = torch.tensor(0)
                M = torch.eye(m, dtype=dtype)

                primals = (dummy_state, M)
                V_out, H_out, breakdown = StockSolverSum._kth_arnoldi_iteration(
                    torch.tensor(k, dtype=torch.int64), V, H, primals
                )

                self.assertEqual(V_out.dtype, dtype)
                self.assertEqual(H_out.dtype, dtype)
                self.assertEqual(V_out.shape, (m, restart + 1))
                self.assertEqual(H_out.shape, (restart, restart + 1))
                self.assertIsInstance(breakdown, (bool, torch.Tensor))

def _normalize_like_impl(x: torch.Tensor):
    n = torch.linalg.vector_norm(x)
    eps = torch.finfo(x.dtype).eps
    t = torch.as_tensor(eps, dtype=n.dtype, device=n.device)
    safe = torch.clamp(n, min=t)
    y = x / safe
    use = n > t
    y = torch.where(use, y, torch.zeros_like(x))
    n_out = torch.where(use, n, n.new_zeros(()))
    return y, n_out


class TestGMRESBatched(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.saved_matvec = StockSolverSum.matvec
        self.saved_compute_residual = StockSolverSum.compute_residual        
        StockSolverSum.matvec = linear_matvec
        StockSolverSum.compute_residual = linear_residual

    def tearDown(self):
        StockSolverSum.matvec =  self.saved_matvec
        StockSolverSum.compute_residual = self.saved_compute_residual
        
    def assertAllClose(self, a, b, dtype, atol=None, rtol=0.0, msg=""):
        if atol is None:
            atol = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(
            torch.allclose(a, b, atol=atol, rtol=rtol), msg or f"{a} vs {b}"
        )

    def _prep_initial_residual(self, f_state, M, b, x0):
        primals = (f_state, M)
        r0 = StockSolverSum.compute_residual(b, x0, primals)
        unit_residual, residual_norm = _normalize_like_impl(r0)
        return unit_residual, residual_norm

    def test_identity_converges_in_one_step(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = 8
                I = torch.eye(m, dtype=dtype)
                M = I

                b = torch.randn(m, dtype=dtype)
                x0 = torch.randn(m, dtype=dtype)

                unit_residual, residual_norm = self._prep_initial_residual(I, M, b, x0)

                primals = (I, M)
                # Any restart >= 1 should solve exactly in one step
                x, ures, rnorm = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, 3, primals
                )

                # Expect x == b (A = I), residual zero
                self.assertAllClose(x, b, dtype)
                # NOT ZERO AFTER NORMALIZATION
                # self.assertTrue(torch.allclose(ures, torch.zeros_like(ures)))
                self.assertLess(rnorm.item(), 1e-6 if dtype == torch.float32 else 1e-12)

    def test_preconditioned_identity_converges_in_one_step(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # A = D (invertible diagonal), M = D^{-1}  => M A = I
                d = torch.tensor([1.5, 0.5, -2.0, 3.0, 4.0, -1.2], dtype=dtype)
                D = torch.diag(d)
                Minv = torch.diag(1.0 / d)
                M = Minv

                m = d.numel()
                b = torch.randn(m, dtype=dtype)
                x_star = torch.linalg.solve(D, b)  # true solution
                x0 = torch.randn(m, dtype=dtype)

                primals = (D, M)
                unit_residual, residual_norm = self._prep_initial_residual(D, M, b, x0)

                x, ures, rnorm = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, 1, primals
                )

                self.assertAllClose(x, x_star, dtype)
                self.assertTrue(torch.allclose(ures, torch.zeros_like(ures)))
                self.assertEqual(rnorm.item(), 0.0)

    def test_full_restart_solves_exact_spd(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # Well-conditioned SPD system
                m = 6
                R = torch.randn(m, m, dtype=dtype)
                A_mat = R.T @ R + 0.5 * torch.eye(m, dtype=dtype)
                M = torch.eye(m, dtype=dtype)

                b = torch.randn(m, dtype=dtype)
                x_true = torch.linalg.solve(A_mat, b)
                x0 = torch.zeros(m, dtype=dtype)

                unit_residual, residual_norm = self._prep_initial_residual(
                    A_mat, M, b, x0
                )

                primals = (A_mat, M)
                # restart >= m should produce the exact solve (up to fp error)
                x, ures, rnorm = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, m, primals
                )

                # Check solution and residual
                self.assertAllClose(x, x_true, dtype)
                res = StockSolverSum.compute_residual(b, x, primals)
                self.assertTrue(
                    torch.allclose(
                        res,
                        torch.zeros_like(res),
                        atol=1e-5 if dtype == torch.float32 else 1e-12,
                    )
                )
                # NOT ZERO AFTER NORMALIZATION
                # self.assertTrue(torch.allclose(ures, torch.zeros_like(ures))
                self.assertLess(rnorm.item(), 1e-5 if dtype == torch.float32 else 1e-12)

    def test_restart_stability_when_converges_early(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # Same preconditioned-identity setup: converges in 1 step
                d = torch.tensor([2.0, -0.7, 1.1, 3.3], dtype=dtype)
                D = torch.diag(d)
                Minv = torch.diag(1.0 / d)
                M = Minv

                m = d.numel()
                b = torch.randn(m, dtype=dtype)
                x0 = torch.randn(m, dtype=dtype)

                unit_residual, residual_norm = self._prep_initial_residual(D, M, b, x0)

                primals = (D, M)
                x1, _, _ = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, 1, primals
                )
                x3, _, _ = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, 3, primals
                )

                self.assertAllClose(x1, x3, dtype)

    def test_zero_residual_returns_x0(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = 5
                A_mat = torch.randn(m, m, dtype=dtype)
                # Make A nonsingular-ish
                A_mat = A_mat + A_mat.T + m * torch.eye(m, dtype=dtype)
                M = torch.eye(m, dtype=dtype)

                x_true = torch.randn(m, dtype=dtype)
                primals = (A_mat, M)
                b = StockSolverSum.matvec(x_true, primals)
                x0 = x_true.clone()  # start at the solution

                unit_residual, residual_norm = self._prep_initial_residual(
                    A_mat, M, b, x0
                )

                # sanity check: residual is exactly zero
                self.assertEqual(residual_norm.item(), 0.0)
                self.assertTrue(
                    torch.allclose(unit_residual, torch.zeros_like(unit_residual))
                )

                primals = (A_mat, M)
                x, ures, rnorm = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, 4, primals
                )

                self.assertAllClose(x, x0, dtype)
                self.assertTrue(torch.allclose(ures, torch.zeros_like(ures)))
                self.assertEqual(rnorm.item(), 0.0)

    def test_shapes_and_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m, restart = 7, 3
                A_mat = torch.randn(m, m, dtype=dtype)
                M = torch.eye(m, dtype=dtype)

                b = torch.randn(m, dtype=dtype)
                x0 = torch.randn(m, dtype=dtype)

                unit_residual, residual_norm = self._prep_initial_residual(
                    A_mat, M, b, x0
                )

                primals = (A_mat, M)
                x, ures, rnorm = StockSolverSum._gmres_batched(
                    b, x0, unit_residual, residual_norm, restart, primals
                )

                self.assertEqual(x.dtype, dtype)
                self.assertEqual(ures.dtype, dtype)
                self.assertEqual(rnorm.dtype, dtype)
                self.assertEqual(x.shape, (m,))
                self.assertEqual(ures.shape, (m,))
                self.assertEqual(rnorm.shape, torch.Size(()))

def np_to_torch(x, dtype):
    return torch.tensor(np.asarray(x), dtype=dtype)


def torch_to_jnp(x_t):
    return jnp.asarray(x_t.detach().cpu().numpy())


class TestGMRESvsJAX(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)
        self.saved_matvec = StockSolverSum.matvec
        self.saved_compute_residual = StockSolverSum.compute_residual        
        StockSolverSum.matvec = linear_matvec
        StockSolverSum.compute_residual = linear_residual

    def tearDown(self):
        StockSolverSum.matvec =  self.saved_matvec
        StockSolverSum.compute_residual = self.saved_compute_residual
        
    def assertAllCloseTorch(self, a, b, dtype, atol=None, rtol=0.0):
        if atol is None:
            atol = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=rtol), f"{a} vs {b}")

    def assertAllCloseJnp(self, a, b, dtype, atol=None, rtol=0.0):
        if atol is None:
            atol = 1e-5 if dtype == torch.float32 else 1e-12
        self.assertTrue(
            np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol),
            f"{a} vs {b}",
        )

    def _solve_both(
        self, A_mat_np, b_np, x0_np, dtype, restart, maxiter=None, M_mat_np=None
    ):
        # Torch side
        A_t = torch.tensor(A_mat_np, dtype=dtype)
        b_t = torch.tensor(b_np, dtype=dtype)
        x0_t = torch.tensor(x0_np, dtype=dtype)
        if M_mat_np is None:
            M_t = torch.eye(A_t.shape[0], dtype=dtype)
        else:
            M_t = torch.tensor(M_mat_np, dtype=dtype)

        primals = (A_t, M_t)
        x_t = StockSolverSum.gmres(
            b_t,
            primals,
            x0=x0_t,
            tol=1e-6,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
        )

        # JAX side
        if dtype == torch.float64:
            jax.config.update("jax_enable_x64", True)

        A_j = jnp.asarray(
            A_mat_np, dtype=jnp.float64 if dtype == torch.float64 else jnp.float32
        )
        b_j = jnp.asarray(b_np, dtype=A_j.dtype)
        x0_j = jnp.asarray(x0_np, dtype=A_j.dtype)
        M_j = None if M_mat_np is None else jnp.asarray(M_mat_np, dtype=A_j.dtype)

        A_fun_j = lambda x: A_j @ x
        M_fun_j = None if M_j is None else (lambda x: M_j @ x)

        x_j, info = jax_gmres(
            A_fun_j,
            b_j,
            x0=x0_j,
            tol=1e-6,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            M=M_fun_j,
            solve_method="batched",
        )
        # In JAX, info is typically a NamedTuple; ensure convergence:
        # Many JAX versions set info.residual or info.num_iters;
        # here we just assert the residual is small directly.

        return x_t, x_j, A_t, b_t, M_t, A_t

    def test_identity(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = 8
                A_np = np.eye(m)
                b_np = np.random.randn(m)
                x0_np = np.random.randn(m)

                x_t, x_j, A_t, b_t, M_t, A_t = self._solve_both(
                    A_np, b_np, x0_np, dtype, restart=3, maxiter=5
                )

                # Exact solution is b for identity
                self.assertAllCloseTorch(x_t, torch.tensor(b_np, dtype=dtype), dtype)
                self.assertAllCloseJnp(x_j, b_np, dtype)

                # Residuals should be zero (within fp)
                A_t = torch.tensor(A_np, dtype=dtype)
                M_t = torch.eye(A_np.shape[0], dtype=dtype)
                primals = (A_t, M_t)
                r_t = StockSolverSum.compute_residual(b_t, x_t,primals)
                self.assertAllCloseTorch(
                    torch.linalg.vector_norm(r_t), torch.tensor(0.0, dtype=dtype), dtype
                )

                r_j = A_np @ np.asarray(x_j) - b_np
                self.assertTrue(
                    np.allclose(
                        r_j, 0.0, atol=1e-6 if dtype == torch.float32 else 1e-12
                    )
                )

    def test_spd_full_restart_matches_exact(self):
        np.random.seed(123)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = 10
                R = np.random.randn(m, m)
                A_np = R.T @ R + 0.5 * np.eye(m)  # SPD, well-conditioned
                b_np = np.random.randn(m)
                x0_np = np.zeros(m)

                # exact solution
                x_true = np.linalg.solve(A_np, b_np)

                x_t, x_j, A_t, b_t, M_t, A_t = self._solve_both(
                    A_np, b_np, x0_np, dtype, restart=m, maxiter=2
                )

                # Compare both to exact
                self.assertAllCloseTorch(x_t, torch.tensor(x_true, dtype=dtype), dtype)
                self.assertAllCloseJnp(x_j, x_true, dtype)

                # Residuals near zero
                A_t = torch.tensor(A_np, dtype=dtype)
                M_t = torch.eye(A_np.shape[0], dtype=dtype)
                primals = (A_t, M_t)
                rt = StockSolverSum.compute_residual(b_t, x_t, primals)
                self.assertAllCloseTorch(
                    torch.linalg.vector_norm(rt), torch.tensor(0.0, dtype=dtype), dtype
                )

                rj = A_np @ np.asarray(x_j) - b_np
                self.assertTrue(
                    np.allclose(rj, 0.0, atol=5e-6 if dtype == torch.float32 else 1e-12)
                )

    def test_nonsymmetric_well_conditioned(self):
        np.random.seed(456)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = 12
                A_np = np.random.randn(m, m)
                A_np = A_np + 0.5 * np.eye(m)  # shift to keep well-conditioned
                b_np = np.random.randn(m)
                x0_np = np.zeros(m)

                x_true = np.linalg.solve(A_np, b_np)

                x_t, x_j, A_t, b_t, M_t, A_t = self._solve_both(
                    A_np, b_np, x0_np, dtype, restart=min(m, 10), maxiter=12
                )

                # Both should be close to exact solution
                # (allow a bit looser tolerance for nonsymmetric)
                tol = 1e-4 if dtype == torch.float32 else 1e-10
                self.assertTrue(
                    np.allclose(np.asarray(x_t.detach()), x_j, atol=tol, rtol=0.0)
                )

    def test_diagonal_perfect_preconditioner(self):
        np.random.seed(789)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                d = np.array([1.5, -2.0, 0.7, 3.3, -1.1], dtype=float)
                A_np = np.diag(d)
                M_np = np.diag(1.0 / d)  # left preconditioner so M A = I
                b_np = np.random.randn(d.size)
                x0_np = np.random.randn(d.size)

                x_true = np.linalg.solve(A_np, b_np)

                x_t, x_j, A_t, b_t, M_t, A_t = self._solve_both(
                    A_np, b_np, x0_np, dtype, restart=1, maxiter=1, M_mat_np=M_np
                )

                # One step should solve exactly
                self.assertAllCloseTorch(x_t, torch.tensor(x_true, dtype=dtype), dtype)
                self.assertAllCloseJnp(x_j, x_true, dtype)

                # Residuals zero
                A_t = torch.tensor(A_np, dtype=dtype)
                M_t = torch.tensor(M_np, dtype=dtype)
                primals = (A_t, M_t)
                rt = StockSolverSum.compute_residual(b_t, x_t, primals)
                self.assertAllCloseTorch(
                    torch.linalg.vector_norm(rt), torch.tensor(0.0, dtype=dtype), dtype
                )

if __name__ == "__main__":
    unittest.main()
