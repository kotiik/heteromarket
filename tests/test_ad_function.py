import importlib
import torch
import unittest

_core = importlib.import_module("heteromarket.core")
GMRESSolver = _core.GMRESSolver
bdot = _core.bdot

# ---------------------------------------------------------------------
# Helper ops used in tests
# ---------------------------------------------------------------------


class AllDiffOp(ExplicitADFunction):
    """
    y0 = x0^2 + x1
    y1 = x1^3
    both outputs differentiable
    saved primals: (x0, x1)
    """

    @staticmethod
    def compute(x0: Tensor, x1: Tensor):
        y0 = x0 * x0 + x1
        y1 = x1 * x1 * x1
        return (y0, y1)

    @staticmethod
    def compute_primals(*inputs: Tensor, outputs):
        x0, x1 = inputs
        return (x0, x1)

    @staticmethod
    def vjp_from_primals(saved, *cotangents, needs_input_grad=None):
        x0, x1 = saved
        v0, v1 = cotangents
        need0 = True if needs_input_grad is None else needs_input_grad[0]
        need1 = True if needs_input_grad is None else needs_input_grad[1]
        gx0 = (2 * x0) * v0 if need0 else None
        gx1 = (1 * v0 + 3 * x1 * x1 * v1) if need1 else None
        return (gx0, gx1)

    @staticmethod
    def jvp_from_primals(saved, *tangents):
        x0, x1 = saved
        t0, t1 = tangents
        if t0 is None:
            t0 = torch.zeros_like(x0)
        if t1 is None:
            t1 = torch.zeros_like(x1)
        dy0 = 2 * x0 * t0 + t1
        dy1 = 3 * x1 * x1 * t1
        return (dy0, dy1)


class NonDiffSecondOp(ExplicitADFunction):
    """
    y0 = x0^2 + x1 (diff)
    y1 = 1_{x1>0} (non-diff)
    saved primals: (x0, x1)
    """

    non_differentiable_output_indices = (1,)

    @staticmethod
    def compute(x0: Tensor, x1: Tensor):
        y0 = x0 * x0 + x1
        y1 = (x1 > 0).to(x1.dtype)
        return (y0, y1)

    @staticmethod
    def compute_primals(*inputs: Tensor, outputs):
        x0, x1 = inputs
        return (x0, x1)

    @staticmethod
    def vjp_from_primals(saved, *cotangents, needs_input_grad=None):
        # Autograd passes one cotangent per output (y0, y1),
        # but y1 is marked non-diff → its cotangent may be None; ignore it.
        x0, x1 = saved

        # Expect (v0, v1) where v1 may be None.
        if len(cotangents) == 0:
            raise RuntimeError("Expected at least one cotangent for y0")
        v0 = cotangents[0]
        if v0 is None:
            v0 = torch.zeros_like(x0)

        need0 = True if needs_input_grad is None else needs_input_grad[0]
        need1 = True if needs_input_grad is None else needs_input_grad[1]

        gx0 = (2 * x0) * v0 if need0 else None
        gx1 = (1 * v0) if need1 else None
        return (gx0, gx1)

    @staticmethod
    def jvp_from_primals(saved, *tangents):
        x0, x1 = saved
        t0, t1 = tangents
        if t0 is None:
            t0 = torch.zeros_like(x0)
        if t1 is None:
            t1 = torch.zeros_like(x1)
        dy0 = 2 * x0 * t0 + t1
        dy1 = torch.zeros_like(x1)  # non-diff output → zero tangent
        return (dy0, dy1)


class CountPrimalsOp(ExplicitADFunction):
    """
    Minimal op to verify compute_primals is invoked exactly once per forward.
    y = x^2
    saved primals: (x,)
    """

    calls = 0

    @staticmethod
    def compute(x: Tensor):
        return x * x

    @staticmethod
    def compute_primals(*inputs: Tensor, outputs):
        CountPrimalsOp.calls += 1
        (x,) = inputs
        return (x,)

    @staticmethod
    def vjp_from_primals(saved, *cotangents, needs_input_grad=None):
        (x,) = saved
        (v,) = cotangents
        need0 = True if needs_input_grad is None else needs_input_grad[0]
        gx = (2 * x) * v if need0 else None
        return (gx,)

    @staticmethod
    def jvp_from_primals(saved, *tangents):
        (x,) = saved
        (t,) = tangents
        if t is None:
            t = torch.zeros_like(x)
        return (2 * x * t,)


# ---------------------------------------------------------------------
# Test cases (unittest)
# ---------------------------------------------------------------------


class TestExplicitADFunction(unittest.TestCase):

    def test_forward_outputs_are_correct(self):
        x0 = torch.tensor(2.0, requires_grad=True)
        x1 = torch.tensor(3.0, requires_grad=True)
        y0, y1 = AllDiffOp.apply(x0, x1)
        self.assertTrue(torch.allclose(y0, torch.tensor(7.0)))
        self.assertTrue(torch.allclose(y1, torch.tensor(27.0)))

    def test_backward_gradients_all_diff(self):
        x0 = torch.tensor(2.0, requires_grad=True)
        x1 = torch.tensor(3.0, requires_grad=True)
        y0, y1 = AllDiffOp.apply(x0, x1)
        loss = y0 * 1.0 + y1 * 2.0
        loss.backward()
        # gx0 = 2*x0*1 = 4
        # gx1 = 1*1 + 3*x1^2*2 = 1 + 54 = 55
        self.assertTrue(torch.allclose(x0.grad, torch.tensor(4.0)))
        self.assertTrue(torch.allclose(x1.grad, torch.tensor(55.0)))

    def test_needs_input_grad_masks_unused_inputs(self):
        x0 = torch.tensor(2.0, requires_grad=True)
        x1 = torch.tensor(3.0, requires_grad=False)  # mask this one
        y0, y1 = AllDiffOp.apply(x0, x1)
        (y0 * 1.0 + y1 * 2.0).backward()
        self.assertTrue(torch.allclose(x0.grad, torch.tensor(4.0)))
        self.assertIsNone(x1.grad)  # masked to None by backward()

    def test_non_differentiable_output_is_marked_and_ignored(self):
        x0 = torch.tensor(2.0, requires_grad=True)
        x1 = torch.tensor(3.0, requires_grad=True)
        y0, y1 = NonDiffSecondOp.apply(x0, x1)

        # Backprop through y0 only (y1 is non-diff)
        (y0 * 1.0).backward()
        self.assertTrue(torch.allclose(x0.grad, torch.tensor(4.0)))
        self.assertTrue(torch.allclose(x1.grad, torch.tensor(1.0)))

        # Using y1 alone in a loss should error (no grad path)
        with self.assertRaises(RuntimeError):
            (y1 * 1.0).backward()

    def test_jvp_formulas_direct_call(self):
        # Test math hook directly (no autograd engine)
        x0 = torch.tensor(2.0)
        x1 = torch.tensor(3.0)
        outputs = AllDiffOp.compute(x0, x1)
        saved = AllDiffOp.compute_primals(x0, x1, outputs=outputs)
        dy0, dy1 = AllDiffOp.jvp_from_primals(
            saved, torch.tensor(1.0), torch.tensor(0.5)
        )
        self.assertTrue(torch.allclose(dy0, torch.tensor(4.5)))  # 2*2*1 + 0.5
        self.assertTrue(torch.allclose(dy1, torch.tensor(13.5)))  # 3*9*0.5

    def test_vjp_formulas_direct_call_and_needs_mask(self):
        x0 = torch.tensor(2.0)
        x1 = torch.tensor(3.0)
        outputs = AllDiffOp.compute(x0, x1)
        saved = AllDiffOp.compute_primals(x0, x1, outputs=outputs)

        v0 = torch.tensor(1.0)
        v1 = torch.tensor(2.0)
        gx0, gx1 = AllDiffOp.vjp_from_primals(
            saved, v0, v1, needs_input_grad=(True, True)
        )
        self.assertTrue(torch.allclose(gx0, torch.tensor(4.0)))
        self.assertTrue(torch.allclose(gx1, torch.tensor(55.0)))

        gx0_m, gx1_m = AllDiffOp.vjp_from_primals(
            saved, v0, v1, needs_input_grad=(True, False)
        )
        self.assertTrue(torch.allclose(gx0_m, torch.tensor(4.0)))
        self.assertIsNone(gx1_m)

    def test_compute_primals_called_once_per_forward(self):
        CountPrimalsOp.calls = 0
        x = torch.tensor(2.0, requires_grad=True)
        y = CountPrimalsOp.apply(x)
        # at this point, setup_context should have run once
        self.assertEqual(CountPrimalsOp.calls, 1)
        # do a backward to ensure gradients flow and no extra primals calls
        (y * 3.0).backward()
        self.assertEqual(CountPrimalsOp.calls, 1)
        self.assertTrue(
            torch.allclose(x.grad, torch.tensor(2 * 2.0 * 3.0))
        )  # d(x^2)*3 = 2x*3
