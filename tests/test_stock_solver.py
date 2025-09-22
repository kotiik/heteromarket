import importlib
import unittest
import torch

_core = importlib.import_module("heteromarket.core")
StockSolver = _core.StockSolver

class TestActiveSetQPSolverInit(unittest.TestCase):
    @staticmethod
    def make_data(B=3, n=4, device='cpu', dtype=torch.float64):
        Q = torch.randn(B, n, n, device=device, dtype=dtype)
        Q = 0.5 * (Q + Q.transpose(-2, -1))  # symmetrize (not required but realistic)
        m = torch.randn(B, n, device=device, dtype=dtype)
        X = torch.full((B, n), 2.0, device=device, dtype=dtype)
        budget = torch.full((B,), 10.0, device=device, dtype=dtype)
        kappa = torch.full((B, ), 0.5, device=device, dtype=dtype)
        theta = torch.full((B, ),  1.5, device=device, dtype=dtype)
        wl = torch.full((B,), 0.5, device=device, dtype=dtype)
        wh = torch.full((B,), 1.5, device=device, dtype=dtype)
        c  = torch.full((B,), 0.1, device=device, dtype=dtype)
        return Q, m, c, X, budget, kappa, theta

    def test_happy_path_cpu(self):
        Q, m, c, X, budget, kappa, theta = self.make_data(B=2, n=3, device='cpu')
        solver = StockSolver(Q, m, c, X, budget, kappa, theta)
        self.assertEqual(solver.B, 2)
        self.assertEqual(solver.n, 3)
        # objects are the same references
        self.assertTrue((solver.Q == Q).all())
        self.assertTrue((solver.m == m).all())
        self.assertTrue((solver.c == c).all())
        self.assertTrue((solver.X == X).all())
        self.assertTrue((solver.budget == budget).all())
        self.assertTrue((solver.kappa == kappa).all())
        self.assertTrue((solver.theta == theta).all())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_device_mismatch_raises(self):
        Q, m, c, X, budget, kappa, theta = self.make_data(B=2, n=3, device='cpu')
        Q_cuda = Q.to('cuda')
        with self.assertRaises(TypeError):
            StockSolver(Q_cuda, m, c, X, budget, kappa, theta)

    def test_non_tensor_raises(self):
        Q, m, c, X, budget, kappa, theta = self.make_data(B=2, n=3)
        with self.assertRaises(TypeError):
            StockSolver(Q, m.tolist(), c, X, budget, kappa, theta)

    def test_Q_shape_errors(self):
        Q, m, c, X, budget, kappa, theta = self.make_data(B=2, n=3)
        with self.assertRaises(ValueError):
            StockSolver(Q[:, :2, :3], m, c, X, budget, kappa, theta)  # not square
        with self.assertRaises(ValueError):
            StockSolver(Q[0], m, c, X, budget, kappa, theta)          # not 3D

    def test_vector_shape_errors(self):
        Q, m, c, X, budget, kappa, theta = self.make_data(B=2, n=3)
        with self.assertRaises(ValueError):
            StockSolver(Q, m[:1], c, X, budget, kappa, theta)     # m wrong B
        with self.assertRaises(ValueError):
            StockSolver(Q, m[:, :2], c, X, budget, kappa, theta)  # m wrong n
        with self.assertRaises(ValueError):
            StockSolver(Q, m, c[:1], X, budget, kappa, theta)     # c wrong B
        with self.assertRaises(ValueError):
            StockSolver(Q, m, c, X, budget[:1], kappa, theta)     # wl wrong B
        with self.assertRaises(ValueError):
            StockSolver(Q, m, c, X, budget, kappa[:1], theta)     # wh wrong B
        with self.assertRaises(ValueError):
            StockSolver(Q, m, c, X, budget, kappa, theta[:1])  # U wrong n

    def test_kappa_theta_sign_violation(self):
        Q, m, c, X, budget, kappa, theta = self.make_data(B=2, n=3)
        kappa_bad = kappa.clone()
        kappa_bad[0] = -3.0
        theta_bad = theta.clone()
        theta_bad[0] = -2.0  # U < L at one coord
        with self.assertRaises(ValueError):
            StockSolver(Q, m, c, X, budget, kappa_bad, theta_bad)

if __name__ == "__main__":
    unittest.main()
