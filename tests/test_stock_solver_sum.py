# test_stock_solver_sum_basic.py
import unittest
import torch
from torch.autograd.functional import jvp

_core = importlib.import_module("heteromarket.core")
StockSolverFunc = _core.StockSolverFunc
StockSolverSum = _core.StockSolverSum


class TestStockSolverSumBasic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dtype = torch.float64
        self.dev = "cpu"

        B, n = 2, 3
        self.B, self.n = B, n

        I = torch.eye(n, dtype=self.dtype, device=self.dev)
        self.Q = (0.1 * I).unsqueeze(0).repeat(B, 1, 1)
        self.m = torch.tensor([[0.02, -0.01, 0.015],
                               [-0.01,  0.01, -0.02]], dtype=self.dtype, device=self.dev)
        self.c = torch.full((B,), 0.1, dtype=self.dtype, device=self.dev)
        self.X = torch.tensor([[0.05, 0.02, 0.03],
                               [0.01, 0.04, 0.02]], dtype=self.dtype, device=self.dev)

        # Wide feasible region -> keep interior, avoid kinks
        self.budget = torch.full((B,), 10.0, dtype=self.dtype, device=self.dev)
        self.kappa  = torch.full((B,), 5.0, dtype=self.dtype, device=self.dev)
        self.theta  = torch.full((B,), 5.0, dtype=self.dtype, device=self.dev)

        self.p  = torch.tensor([1.2, 1.5, 1.8], dtype=self.dtype, device=self.dev)
        self.x0 = torch.zeros((B, n), dtype=self.dtype, device=self.dev)

    def _apply(self, Q, m, c, X, budget, kappa, theta, p, x0):
        return StockSolverSum.apply(Q, m, c, X, budget, kappa, theta, p, x0)

    def test_forward_runs(self):
        y = self._apply(
            self.Q,
            self.m,
            self.c,
            self.X,
            self.budget,
            self.kappa,
            self.theta,
            self.p,
            self.x0,
        )
        self.assertEqual(tuple(y.shape), (self.n,))
        self.assertTrue(torch.isfinite(y).all())

    def test_backward_runs(self):
        Q = self.Q.clone().requires_grad_(True)
        m = self.m.clone().requires_grad_(True)
        c = self.c.clone().requires_grad_(True)
        X = self.X.clone().requires_grad_(True)
        budget = self.budget.clone().requires_grad_(True)
        kappa  = self.kappa.clone().requires_grad_(True)
        theta  = self.theta.clone().requires_grad_(True)
        p      = self.p.clone().requires_grad_(True)
        x0     = self.x0.clone().requires_grad_(True)

        y = self._apply(Q, m, c, X, budget, kappa, theta, p, x0)  # (n,)
        loss = y.sum()
        loss.backward()

        # Just sanity: some grads exist and have right shapes
        self.assertIsInstance(p.grad, torch.Tensor)
        self.assertEqual(tuple(p.grad.shape), (self.n,))
        self.assertIsInstance(m.grad, torch.Tensor)
        self.assertEqual(tuple(m.grad.shape), (self.B, self.n))

    def test_backward_with_vector_vjp_shape(self):
        Q = self.Q.clone().requires_grad_(True)
        m = self.m.clone().requires_grad_(True)
        c = self.c.clone().requires_grad_(True)
        X = self.X.clone().requires_grad_(True)
        budget = self.budget.clone().requires_grad_(True)
        kappa  = self.kappa.clone().requires_grad_(True)
        theta  = self.theta.clone().requires_grad_(True)
        p      = self.p.clone().requires_grad_(True)
        x0     = self.x0.clone().requires_grad_(True)

        y = self._apply(Q, m, c, X, budget, kappa, theta, p, x0)  # (n,)
        v = torch.randn_like(y)
        (v * y).sum().backward()

        self.assertIsInstance(Q.grad, torch.Tensor)
        self.assertEqual(tuple(Q.grad.shape), (self.B, self.n, self.n))
        self.assertIsInstance(X.grad, torch.Tensor)
        self.assertEqual(tuple(X.grad.shape), (self.B, self.n))

    def test_jvp_runs(self):
        primals = (
            self.Q.clone(), self.m.clone(), self.c.clone(), self.X.clone(),
            self.budget.clone(), self.kappa.clone(), self.theta.clone(),
            self.p.clone(), self.x0.clone()
        )
        # Small tangents (no need for accuracy checks here)
        tangents = tuple(
            torch.zeros_like(t) if t.dim() == 3 else torch.zeros_like(t)
            for t in primals
        )
        # add a small nonzero direction on p and X
        tangents = list(tangents)
        tangents[3] = torch.full_like(self.X, 1e-5)  # dX
        tangents[7] = torch.full_like(self.p, 1e-5)  # dp
        tangents = tuple(tangents)

        def f(*args):
            return self._apply(*args)

        y0, dy = jvp(f, primals, tangents)
        self.assertEqual(tuple(y0.shape), (self.n,))
        self.assertEqual(tuple(dy.shape), (self.n,))
        self.assertTrue(torch.isfinite(y0).all())
        self.assertTrue(torch.isfinite(dy).all())


if __name__ == "__main__":
    unittest.main()
