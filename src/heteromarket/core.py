import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as functional
from torch import ops

import numpy as np
import math
from typing import NamedTuple, Callable, Tuple


class SolverState(NamedTuple):
    """
    Internal state container for the batched active‑set quadratic‑program solver.

    The solver operates on a batch of QP instances, so every tensor stored in
    this ``NamedTuple`` has a leading batch dimension ``B``.  At each iteration
    the solver updates the fields to reflect the current approximation of the
    solution and the status of the active constraints.

    Attributes
    ----------
    x : torch.Tensor
        Current primal estimate of the solution vector.
        Shape: ``(B, n)`` where ``n`` is the number of decision variables.
        Each row corresponds to the current approximation for a single batch
        instance.

    active_comp : torch.Tensor
        Boolean mask indicating which box (bound) constraints are currently
        active for each variable.
        Shape: ``(B, n)`` – ``True`` means the corresponding component of ``x``
        is clamped to its bound.

    active_values : torch.Tensor
        Values of the active box constraints (the right‑hand side of the bound)
        for each variable.
        Shape: ``(B, n)`` – entries are meaningful only where ``active_comp`` is
        ``True``.

    active_budget : torch.Tensor
        Boolean flag that tells whether the linear “budget” constraint is active
        for each batch element.
        Shape: ``(B,)`` – ``True`` means the budget constraint participates in
        the current KKT system.

    budget_w : torch.Tensor
        Right‑hand side (weight) of the budget constraint when it is active.
        Shape: ``(B,)`` – ignored for batches where ``active_budget`` is ``False``.

    done : torch.Tensor
        Convergence indicator for each batch instance.
        Shape: ``(B,)`` – ``True`` means the corresponding problem has satisfied
        optimality conditions and will no longer be updated.

    Notes
    -----
    * The solver updates all fields in‑place (or returns a new ``SolverState``
      instance) after each iteration.
    * Because the class inherits from ``NamedTuple``, the fields are immutable;
      creating a new state is cheap and safe for functional‑style updates.
    * All tensors are expected to reside on the same device (CPU/GPU) and share
      the same dtype for consistency across operations.
    """

    x: torch.Tensor  # (B, n)
    active_comp: torch.Tensor  # (B, n), boolean
    active_values: torch.Tensor  # (B, n)
    active_budget: torch.Tensor  # (B, ), boolean
    budget_w: torch.Tensor  # (B, )
    done: torch.Tensor  # (B, ), boolean


class SoverParameters(NamedTuple):
    """
    Parameter bundle for a batched quadratic‑program (QP) solved by the active‑set
    solver.

    The QP has the form

        minimize   xᵀ Q x + mᵀ x
        subject to L ≤ x ≤ U                     (box constraints)
                   wl ≤ pᵀ x ≤ wh               (single linear “budget” constraint)

    All tensors are batched; the leading dimension ``B`` indexes independent
    problem instances that are solved simultaneously.

    Attributes
    ----------
    Q : torch.Tensor
        Positive‑definite quadratic coefficient matrices.
        Shape: ``(B, n, n)`` where ``n`` is the number of decision variables.
        Each slice ``Q[b]`` defines the quadratic term for batch ``b``.
        Note that solver does not check for

    m : torch.Tensor
        Linear coefficients of the objective.
        Shape: ``(B, n)`` – ``m[b]`` corresponds to the linear term for batch ``b``.

    c : torch.Tensor
        Commission. The derivative at solution point must be greater than this value,
        otherwise components with smaller derivative will be set to 0.
        Shape: ``(B,)`` – one value per batch element.

    wl : torch.Tensor
        Lower bound of the scalar budget (linear) constraint ``pᵀ x``.
        Shape: ``(B,)`` – one value per batch element.

    wh : torch.Tensor
        Upper bound of the budget constraint.
        Shape: ``(B,)`` – one value per batch element.

    L : torch.Tensor
        Lower bounds for the box constraints on each variable.
        Shape: ``(B, n)`` – ``L[b, i]`` is the lower limit for variable ``i`` in
        batch ``b``.

    U : torch.Tensor
        Upper bounds for the box constraints.
        Shape: ``(B, n)`` – analogous to ``L`` but for the upper limits.

    p : torch.Tensor
        Coefficients of the linear budget constraint ``pᵀ x``.
        Shape: ``(B, n)`` – each row defines the linear combination of variables
        that forms the budget expression for the corresponding batch.

    Notes
    -----
    * The class inherits from ``NamedTuple``; therefore the fields are immutable.
      Updating parameters for a new solve typically involves constructing a fresh
      ``SolverParameters`` instance.
    * All tensors should share the same device (CPU/GPU) and dtype to avoid
      runtime mismatches during matrix operations.
    * The quadratic matrix ``Q`` is assumed symmetric and positive definite;
      callers should ensure this property for correct solver behaviour.
    """

    Q: torch.Tensor  # (B, n, n)
    m: torch.Tensor  # (B, n)
    c: torch.Tensor  # (B, )
    wl: torch.Tensor  # (B, )
    wh: torch.Tensor  # (B, )
    L: torch.Tensor  # (B, n)
    U: torch.Tensor  # (B, n)
    p: torch.Tensor  # (B, n)


class SolverPrimals(NamedTuple):
    """
    Intermidiate values for backpropogation of gradient
    """

    H: torch.Tensor
    p_eff: torch.Tensor
    Lc: torch.Tensor
    sol: torch.Tensor
    x_eq: torch.Tensor
    y: torch.Tensor
    p_dot_xeq: torch.Tensor
    denom_proj: torch.Tensor
    alpha: torch.Tensor


# ----------------------------------------------------------------------
# Batched dot product
# ----------------------------------------------------------------------
def bdot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the element‑wise dot product for a batch of vectors.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(B, N)`` containing the first set of vectors.
    y : torch.Tensor
        Tensor of shape ``(B, N)`` containing the second set of vectors.
        Must be broadcastable to the shape of ``x``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, )`` where each entry is ``∑_k x[..., k] * y[..., k]``—
        i.e., the dot product computed along the second dimension.

    """
    return (x * y).sum(dim=-1)


# ----------------------------------------------------------------------
# Batched quadratic form  xᵀ Q x
# ----------------------------------------------------------------------
def bquad(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the quadratic form ``xᵀ Q x`` for a batch of vectors.

    Parameters
    ----------
    Q : torch.Tensor
        Tensor of shape ``(B, N, N)`` representing a batch of symmetric
        (typically positive‑definite) matrices.
    x : torch.Tensor
        Tensor of shape ``(B, N)`` containing the vectors for each batch entry.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B,)`` where each element is the scalar result of
        ``x[b]ᵀ Q[b] x[b]`` for the corresponding batch index ``b``.

    """
    return torch.einsum("bi,bij,bj->b", x, Q, x)


# ----------------------------------------------------------------------
# Batched matrix‑vector product
# ----------------------------------------------------------------------
def bmv(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Perform a batched matrix‑vector multiplication.

    Parameters
    ----------
    M : torch.Tensor
        Tensor of shape ``(B, M, N)`` containing a batch of matrices.
    x : torch.Tensor
        Tensor of shape ``(B, N)`` containing a batch of column vectors.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, M)`` where each slice ``result[b]`` equals
        ``M[b] @ x[b]``.

    """
    return (M @ x.unsqueeze(-1)).squeeze(-1)


class ActiveSetQPFunc(torch.autograd.Function):
    @staticmethod
    def _unpack_state_params(args: tuple):
        n_state = len(SolverState._fields)
        return SolverState(*args[:n_state]), SoverParameters(*args[n_state:])

    @staticmethod
    def _solve_under_active_int(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        active_comp: torch.Tensor,  # (B, n)
        active_values: torch.Tensor,  # (B, n)
        active_budget: torch.Tensor,  # (B, n)
        budget_w: torch.Tensor,  # (B,)
    ):
        # ---- Patch rows/cols for pinned variables ----
        # Zero rows and columns at active coordinates
        H = 2 * Q
        Q1 = H.masked_fill(active_comp.unsqueeze(-1), 0.0)  # zero rows
        Q1 = Q1.masked_fill(active_comp.unsqueeze(-2), 0.0)  # zero columns
        Q1 = Q1 + torch.diag_embed(active_comp.to(Q1.dtype))

        # Move fixed-vars contribution to RHS for FREE rows:
        # v1_free -= H * x_fixed
        # and set RHS on ACTIVE rows to the fixed values
        v1 = torch.where(
            active_comp,
            -active_values,
            m + bmv(H, active_values),
        )

        # Prepare for projection onto active budget plane (if any)
        p_eff = torch.where(active_comp, 0.0, p)

        # Build the RHS matrix with two columns: [-v1, p_eff]
        # and solve patched system Q1 * [x_eq, y] = [-v1, p_eff] via Cholesky
        Lc = torch.linalg.cholesky(Q1)  # (B, n, n), lower-tri
        rhs = torch.stack((-v1, p_eff), dim=-1)
        sol = torch.cholesky_solve(rhs, Lc)  # (B, n, 2)

        x_eq = sol[..., 0]  # (B, n)  solves Q1 x_eq = -v1
        y = sol[..., 1]  # (B, n)  solves Q1 y    =  p_eff

        # Project onto active budget plane (if any), w/o moving pinned coords
        denom_proj = bdot(p_eff, y).clamp_min(1e-30)
        p_dot_xeq = bdot(p, x_eq)  # (B,)
        # don't move pinned coords
        alpha = torch.where(
            active_budget, (budget_w - p_dot_xeq) / denom_proj, 0.0
        )
        return SolverPrimals(
            H, p_eff, Lc, sol, x_eq, y, p_dot_xeq, denom_proj, alpha
        )

    @staticmethod
    def _solve_under_active(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        active_comp: torch.Tensor,  # (B, n)
        active_values: torch.Tensor,  # (B, n)
        active_budget: torch.Tensor,  # (B, n)
        budget_w: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        primals = ActiveSetQPFunc._solve_under_active_int(
            Q, m, p, active_comp, active_values, active_budget, budget_w
        )
        return primals.x_eq + primals.alpha.unsqueeze(-1) * primals.y

    @staticmethod
    def _line_search_to_constraints(
        x: torch.Tensor,  # (B, n) starting point (must be feasible)
        x_eq_proj: torch.Tensor,  # (B, n) target point
        p: torch.Tensor,  # (B, n)
        wl: torch.Tensor,  # (B, )
        wh: torch.Tensor,  # (B, )
        L: torch.Tensor,  # (B, n)
        U: torch.Tensor,  # (B, n)
        active_comp: torch.Tensor,  # (B, n)
        active_budget: torch.Tensor,  # (B, n)
    ) -> torch.Tensor:
        """
        Make the largest feasible step from x toward x_eq_proj while
        respecting box bounds [L, U], the zero-crossing constraint, and
        the budget planes p^T x ∈ {wl, wh}.
        Returns:
            x_new: (B, n)
        """
        eps = 1e-10
        # Direction to the equality/active-set solution
        delta = x_eq_proj - x

        # 1) distance to box constraints
        alpha_to_U = torch.where(
            (delta > 0.0) & ~active_comp, (U - x) / delta, 1.0
        ).amin(dim=-1)
        alpha_to_L = torch.where(
            (delta < 0.0) & ~active_comp, (L - x) / delta, 1.0
        ).amin(dim=-1)

        # 2) line search toward budget planes
        p_dot_x = bdot(p, x)  # (B,)
        p_dot_delta = bdot(p, delta)  # (B,)

        mask = (torch.abs(p_dot_delta) < eps) | active_budget
        alpha_budget_low = (wl - p_dot_x) / p_dot_delta
        alpha_budget_high = (wh - p_dot_x) / p_dot_delta
        alpha_budget_low = torch.where(
            mask | (alpha_budget_low < -eps) | (p_dot_delta > 0.0), 1.0, alpha_budget_low
        )
        alpha_budget_high = torch.where(
            mask | (alpha_budget_high < -eps) | (p_dot_delta < 0.0), 1.0, alpha_budget_high
        )
        # aggregate step
        alpha = torch.minimum(alpha_to_U, alpha_to_L)
        alpha = torch.minimum(alpha, alpha_budget_low)
        alpha = torch.minimum(alpha, alpha_budget_high).clamp(min=0.0, max=1.0)
        x_new = x + alpha.unsqueeze(-1) * delta

        return x_new

    @staticmethod
    def _compute_projected_gradients(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        x_new: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        active_comp: torch.Tensor,  # (B, n)
        active_budget: torch.Tensor,  # (B, n)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient g (projected onto the budget tangent if a budget
        constraint is active) and its component along the budget normal.
        Returns:
            g: (B, n)
            g_along_p: (B,)
        """
        H = 2 * Q
        # 1) raw gradient ∇f = H x + m, where H = 2Q is in self.H
        grad = bmv(H, x_new) + m  # (B, n)

        # 2) project onto budget tangent only if a budget side is active
        denom = bdot(p, p).clamp_min(1e-30)  # (B,)
        proj_term = (bdot(p, grad) / denom).unsqueeze(-1) * p  # (B, n)

        g = torch.where(active_budget.unsqueeze(-1), grad - proj_term, grad)

        # 3) gradient component along the budget normal p,
        #    restricted to free coords
        p_eff = torch.where(active_comp, 0.0, p)
        g_along_p = bdot(p_eff, grad)  # (B,)

        return g, g_along_p

    @staticmethod
    def _compute_active_sets(
        x_new: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        g: torch.Tensor,  # (B, n)  projected-or-raw per your logic
        g_along_p: torch.Tensor,  # (B,)
        Q: torch.Tensor,  # (B, n, n)
        c: torch.Tensor,  # (B, )
        wl: torch.Tensor,  # (B, )
        wh: torch.Tensor,  # (B, )
        L: torch.Tensor,  # (B, n)
        U: torch.Tensor,  # (B, n)
        active_comp: torch.Tensor,  # (B, n)
        active_budget: torch.Tensor,  # (B,)
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Update active-set masks based on KKT sign rules at the new point.
        Returns:
            active_low, active_high, active_zero, active_budg_low, active_budg_high
        """
        eps = 1e-7

        # 1) box constraints
        # keep lower if g_i >= 0; keep upper if g_i <= 0
        low_keep = ((g >= -eps) | ~active_comp) & (
            x_new <= (L + eps)
        )  # (B, n)
        high_keep = ((g <= eps) | ~active_comp) & (
            x_new >= (U - eps)
        )  # (B, n)

        # 2) zero constraint
        Q_diag = torch.diagonal(Q, dim1=-2, dim2=-1)
        zero_keep = torch.abs(g - Q_diag * x_new) <= c.unsqueeze(-1)  # (B, n)

        # 3) budget constraints
        p_dot_xnew = bdot(p, x_new)  # (B,)

        # keep lower budget if g·p >= 0; keep upper budget if g·p <= 0
        budg_low_keep = ((g_along_p >= -eps) | ~active_budget) & (
            p_dot_xnew <= (wl + eps)
        )
        budg_high_keep = ((g_along_p <= eps) | ~active_budget) & (
            p_dot_xnew >= (wh - eps)
        )
        return low_keep, high_keep, zero_keep, budg_low_keep, budg_high_keep

    @staticmethod
    def _update_active_sets(
        state: SolverState, params: SoverParameters
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g, g_along_p = ActiveSetQPFunc._compute_projected_gradients(
            params.Q,
            params.m,
            state.x,
            params.p,
            state.active_comp,
            state.active_budget,
        )
        # Apply KKT sign rules to (de)activate constraints where we hit
        (
            low_keep,
            high_keep,
            zero_keep,
            budg_low_keep,
            budg_high_keep,
        ) = ActiveSetQPFunc._compute_active_sets(
            state.x,
            params.p,
            g,
            g_along_p,
            params.Q,
            params.c,
            params.wl,
            params.wh,
            params.L,
            params.U,
            state.active_comp,
            state.active_budget,
        )
        # update only for batches that hit a constraint this iteration
        active_comp = torch.where(
            (~state.done).unsqueeze(-1),
            low_keep | high_keep | zero_keep,
            state.active_comp,
        )

        # for active_values components which didn't hit a constraint
        # will be ignored
        active_values = torch.where(
            low_keep, params.L, torch.where(high_keep, params.U, 0.0)
        )
        active_budget = torch.where(
            ~state.done, budg_low_keep | budg_high_keep, state.active_budget
        )
        budget_w = torch.where(budg_low_keep, params.wl, params.wh)  # (B,)

        return active_comp, active_values, active_budget, budget_w

    @staticmethod
    def _setup_loop(
        Q: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        wl: torch.Tensor,
        wh: torch.Tensor,
        L: torch.Tensor,
        U: torch.Tensor,
        p: torch.Tensor,
        x0: torch.Tensor,
    ) -> SolverState:
        # Initialize active sets
        active_comp = torch.zeros_like(U, dtype=torch.bool)
        active_values = torch.zeros_like(U)
        active_budget = torch.zeros_like(wh, dtype=torch.bool)
        budget_w = torch.zeros_like(wh)

        # Mark infeasible batches as done
        # for the rest we never exit after the first iteraction
        done = (
            bdot(p, L).gt(wh) | bdot(p, U).lt(wl) | torch.any(p <= 0.0, dim=-1)
        )

        # 1) Optimal point under current active constraints
        # (incl. budget projection if active)
        x_eq_proj = ActiveSetQPFunc._solve_under_active(
            Q, m, p, active_comp, active_values, active_budget, budget_w
        )
        # 2) ActiveSetQPFunc feasible step toward that point
        x = ActiveSetQPFunc._line_search_to_constraints(
            x0, x_eq_proj, p, wl, wh, L, U, active_comp, active_budget
        )
        return SolverState(
            x, active_comp, active_values, active_budget, budget_w, done
        )

    # takes SolverState and SolverParams in unpacked form
    @staticmethod
    def exit_cond(*args):
        state, _ = ActiveSetQPFunc._unpack_state_params(args)
        return (~state.done).any()

    # takes SolverState and SolverParams in unpacked form
    @staticmethod
    def main_loop(*args):
        state, params = ActiveSetQPFunc._unpack_state_params(args)
        tol = 1e-6
        (
            active_comp_out,
            active_values_out,
            active_budget_out,
            budget_w_out,
        ) = ActiveSetQPFunc._update_active_sets(state, params)
        x_eq_proj = ActiveSetQPFunc._solve_under_active(
            params.Q,
            params.m,
            params.p,
            active_comp_out,
            active_values_out,
            active_budget_out,
            budget_w_out,
        )
        x_new = ActiveSetQPFunc._line_search_to_constraints(
            state.x,
            x_eq_proj,
            params.p,
            params.wl,
            params.wh,
            params.L,
            params.U,
            active_comp_out,
            active_budget_out,
        )
        done_out = (
            state.done
            # new optimal point = current
            | (torch.linalg.vector_norm(x_eq_proj - state.x, dim=-1) < tol)
            # we made full step to optimal point
            | (torch.linalg.vector_norm(x_new - x_eq_proj, dim=-1) < tol)
            # there is no progress
            | (torch.linalg.vector_norm(x_new - state.x, dim=-1) < tol)
        )
        return (
            # do not update done components
            torch.where(state.done.unsqueeze(-1), state.x, x_new),
            active_comp_out,
            active_values_out,
            active_budget_out,
            budget_w_out,
            done_out,
        )

    @staticmethod
    def _finalize_active_sets(
        x: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        wl: torch.Tensor,  # (B, )
        wh: torch.Tensor,  # (B, )
        L: torch.Tensor,  # (B, n)
        U: torch.Tensor,  # (B, n)
    ) -> tuple[
        torch.Tensor,  # x (B, n)
        torch.Tensor,  # active_comp (B, n)
        torch.Tensor,  # active_values (B, n)
        torch.Tensor,  # active_budget (B, )
        torch.Tensor,  # budget_w  (B, )
    ]:
        eps = 1e-7
        active_comp = (x <= L + eps) | (x >= U + eps)
        active_values = torch.where(
            active_comp, torch.where(x <= L + eps, L, U), 0.0
        )
        active_budget = (torch.abs(bdot(x, p) - wl) < eps) | (
            torch.abs(bdot(x, p) - wh) < eps
        )
        budget_w = torch.where(
            active_budget,
            torch.where(
                torch.abs(bdot(x, p) - wl) < eps,
                wl,
                wh,
            ),
            0.0,
        )
        return x, active_comp, active_values, active_budget, budget_w

    @staticmethod
    def forward(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        c: torch.Tensor,  # (B, )
        wl: torch.Tensor,  # (B, )
        wh: torch.Tensor,  # (B, )
        L: torch.Tensor,  # (B, n)
        U: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        x0: torch.Tensor,  # (B, n)
    ) -> tuple[
        torch.Tensor,  # x (B, n)
        torch.Tensor,  # active_comp (B, n)
        torch.Tensor,  # active_values (B, n)
        torch.Tensor,  # active_budget (B, )
        torch.Tensor,  # budget_w  (B, )
    ]:
        state = ActiveSetQPFunc._setup_loop(Q, m, c, wl, wh, L, U, p, x0)
        (
            x,
            active_comp,
            active_values,
            active_budget,
            budget_w,
            _,
        ) = ops.higher_order.while_loop(
            ActiveSetQPFunc.exit_cond,
            ActiveSetQPFunc.main_loop,
            tuple(state),
            (
                Q,
                m,
                c,
                wl,
                wh,
                L,
                U,
                p,
            ),
        )

        # Mark infeasible batches
        x = torch.where(
            (
                bdot(p, L).le(wh)
                & bdot(p, U).ge(wl)
                & torch.all(p > 0.0, dim=-1)
            ).unsqueeze(-1),
            x,
            torch.nan,
        )
        return ActiveSetQPFunc._finalize_active_sets(x, p, wl, wh, L, U)

    @staticmethod
    def setup_context(ctx, inputs: tuple, output: tuple):
        Q, m, _, wl, wh, L, U, p, _ = inputs
        x, active_comp, active_values, active_budget, budget_w = output

        # Non-diff aux outputs:
        ctx.mark_non_differentiable(
            active_comp, active_values, active_budget, budget_w
        )

        # stash as attributes for JVP path (functorch may not populate saved_tensors)
        ctx.saved_for_jvp = (
            Q,
            m,
            wl,
            wh,
            L,
            U,
            p,
            active_comp,
            active_values,
            active_budget,
            budget_w,
        )

        # Save for backward (VJP path)
        ctx.save_for_backward(*ctx.saved_for_jvp)

    @staticmethod
    def _grad_solve(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        p: torch.Tensor,  # (B, n)
        active_comp: torch.Tensor,  # (B, n), bool
        active_values: torch.Tensor,  # (B, n)
        active_budget: torch.Tensor,  # (B,) or (B, n) boolean
        budget_w: torch.Tensor,  # (B,)
        grad_x: torch.Tensor,  # (B, n) upstream gradient
    ):
        (
            H,
            p_eff,
            Lc,
            sol,
            x_eq,
            y,
            p_dot_xeq,
            denom_proj,
            alpha,
        ) = ActiveSetQPFunc._solve_under_active_int(
            Q, m, p, active_comp, active_values, active_budget, budget_w
        )

        # ---- Backward accumulation ----
        # Initialize grads
        grad_Q = torch.zeros_like(Q)
        grad_m = torch.zeros_like(m)
        grad_p = torch.zeros_like(p)
        grad_av = torch.zeros_like(active_values)
        grad_bw = torch.zeros_like(budget_w)

        # x = x_eq + alpha * y
        grad_x_eq = grad_x.clone()
        grad_y = alpha.unsqueeze(-1) * grad_x
        grad_alpha = torch.where(active_budget, bdot(grad_x, y), 0.0)

        denom_safe = torch.abs(denom_proj) > 1e-15
        # d alpha / d budget_w
        grad_bw += torch.where(denom_safe, grad_alpha / denom_proj, 0.0)

        # d alpha / d (p·x_eq)
        grad_pdot = torch.where(denom_safe, -grad_alpha / denom_proj, 0.0)
        grad_p += grad_pdot.unsqueeze(-1) * x_eq
        grad_x_eq += grad_pdot.unsqueeze(-1) * p

        # d alpha / d denom_raw  (respect clamp_min)
        g_denom = torch.where(
            denom_safe,
            -grad_alpha * (budget_w - p_dot_xeq) / (denom_proj * denom_proj),
            0.0,
        )
        # denom_raw = <p_eff, y>
        grad_p_eff_from_denom = g_denom.unsqueeze(-1) * y
        grad_y += g_denom.unsqueeze(-1) * p_eff

        # ---- Backprop through the two solves together: sol = Q1^{-1} rhs ----
        # G = dL/d sol
        G = torch.stack((grad_x_eq, grad_y), dim=-1)  # (B, n, 2)

        # grad wrt rhs: Q1^{-1} G
        grad_rhs = torch.cholesky_solve(G, Lc)  # (B, n, 2)

        # grad wrt Q1: - sym(Q1^{-1} G sol^T)
        GXt = G @ sol.transpose(-1, -2)  # (B, n, n)
        S = torch.cholesky_solve(GXt, Lc)  # (B, n, n)
        grad_Q1 = -0.5 * (S + S.transpose(-1, -2))

        # Unpack rhs contributions
        grad_v1 = -grad_rhs[..., 0]  # from col 0
        grad_p_eff = grad_rhs[..., 1] + grad_p_eff_from_denom

        # p_eff = where(active_comp, 0, p)
        grad_p += torch.where(active_comp, 0.0, grad_p_eff)

        # v1 = where(active_comp, -active_values, m + H @ active_values)
        gv = torch.where(active_comp, 0.0, grad_v1)  # grads on free rows only

        # to m
        grad_m += gv

        # to active_values
        #   - from active rows: v1_i = -active_values_i
        #   - from free rows via H @ active_values
        grad_av += torch.where(active_comp, -grad_v1, 0.0) + bmv(
            H.transpose(-1, -2), gv
        )

        # to H from (H @ active_values) on free rows: outer(gv, active_values)
        grad_H_from_v1 = torch.einsum("bi,bj->bij", gv, active_values)

        # Map grad_Q1 -> H only on free-free block (masked rows/cols do not depend on H)
        grad_H_from_Q1 = grad_Q1.masked_fill(active_comp.unsqueeze(-1), 0.0)
        grad_H_from_Q1 = grad_H_from_Q1.masked_fill(
            active_comp.unsqueeze(-2), 0.0
        )

        grad_H = grad_H_from_v1 + grad_H_from_Q1

        # H = 2Q
        grad_Q = 2.0 * grad_H

        return grad_Q, grad_m, grad_p, grad_av, grad_bw

    @staticmethod
    def backward(ctx, grad_x, *_unused_grads):
        (
            Q,
            m,
            wl,
            wh,
            L,
            U,
            p,
            active_comp,
            active_values,
            active_budget,
            budget_w,
        ) = ctx.saved_tensors
        grad_Q, grad_m, grad_p, grad_av, grad_bw = ActiveSetQPFunc._grad_solve(
            Q,
            m,
            p,
            active_comp,
            active_values,
            active_budget,
            budget_w,
            grad_x,
        )
        grad_L = torch.where(
            active_comp & (active_values == L),
            grad_av,
            torch.zeros_like(grad_av),
        )
        grad_U = torch.where(
            active_comp & (active_values == U),
            grad_av,
            torch.zeros_like(grad_av),
        )
        grad_wl = torch.where(
            active_budget & (budget_w == wl),
            grad_bw,
            torch.zeros_like(grad_bw),
        )
        grad_wh = torch.where(
            active_budget & (budget_w == wh),
            grad_bw,
            torch.zeros_like(grad_bw),
        )
        return (
            grad_Q if ctx.needs_input_grad[0] else None,
            grad_m if ctx.needs_input_grad[1] else None,
            None,
            grad_wl if ctx.needs_input_grad[3] else None,
            grad_wh if ctx.needs_input_grad[4] else None,
            grad_L if ctx.needs_input_grad[5] else None,
            grad_U if ctx.needs_input_grad[6] else None,
            grad_p if ctx.needs_input_grad[7] else None,
            None,
        )

    @torch.no_grad()
    @staticmethod
    def jvp(ctx, dQ, dm, dc, dwl, dwh, dL, dU, dp, dx0):
        saved_primals = ActiveSetQPFunc._process_primals(ctx.saved_for_jvp)
        return (
            ActiveSetQPFunc._jvp_int(
                saved_primals, dQ, dm, dc, dwl, dwh, dL, dU, dp, dx0
            ),
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _process_primals(saved_vars):
        (
            Q,
            m,
            wl,
            wh,
            L,
            U,
            p,
            active_comp,
            active_values,
            active_budget,
            budget_w,
        ) = saved_vars
        primals = ActiveSetQPFunc._solve_under_active_int(
            Q, m, p, active_comp, active_values, active_budget, budget_w
        )
        return (
            *saved_vars,
            primals.H,
            primals.p_eff,
            primals.Lc,
            primals.x_eq.clone(),  # DO I NEED TO CLONE THEM?
            primals.y.clone(),
            primals.p_dot_xeq,
            primals.denom_proj,
            primals.alpha,
        )

    @staticmethod
    def _jvp_int(saved_primals, dQ, dm, dc, dwl, dwh, dL, dU, dp, dx0):
        # saved and processed primals
        (
            Q,
            m,
            wl,
            wh,
            L,
            U,
            p,
            active_comp,
            active_values,
            active_budget,
            budget_w,
            H,
            p_eff,
            Lc,
            x_eq,
            y,
            p_dot_xeq,
            denom_proj,
            alpha,
        ) = saved_primals
        # compute dav = d_active_values and dbw = d_budget_w
        # from dwl, dwh, dL, dU
        dav = torch.where(
            active_comp,
            torch.where(active_values == L, dL, dU),
            torch.zeros_like(L),
        )
        dbw = torch.where(
            active_budget,
            torch.where(budget_w == wl, dwl, dwh),
            torch.zeros_like(wl),
        )

        dH = 2.0 * dQ
        dQ1 = dH.masked_fill(active_comp.unsqueeze(-1), 0.0)
        dQ1 = dQ1.masked_fill(active_comp.unsqueeze(-2), 0.0)

        # RHS terms
        dv1 = torch.where(
            active_comp,
            -dav,
            dm + bmv(dH, active_values) + bmv(H, dav),
        )

        dp_eff = torch.where(active_comp, 0.0, dp)

        # Forward lin. systems for tangents:
        # Q1 dx_eq + dQ1 x_eq = -dv1   =>   dx_eq = Q1^{-1} ( -dv1 - dQ1 x_eq )
        rhs_dx = -dv1 - bmv(dQ1, x_eq)
        dx_eq = torch.cholesky_solve(rhs_dx.unsqueeze(-1), Lc).squeeze(-1)

        # Q1 dy + dQ1 y = dp_eff       =>   dy = Q1^{-1} ( dp_eff - dQ1 y )
        rhs_dy = dp_eff - bmv(dQ1, y)
        dy = torch.cholesky_solve(rhs_dy.unsqueeze(-1), Lc).squeeze(-1)

        # Differentiate alpha (don’t backprop through clamped denom)
        d_pdot = bdot(dp, x_eq) + bdot(p, dx_eq)
        d_denom = bdot(dp_eff, y) + bdot(p_eff, dy)

        dalpha = torch.where(
            active_budget & (torch.abs(denom_proj) > 1e-15),
            (dbw - d_pdot) / denom_proj
            - ((budget_w - p_dot_xeq) / (denom_proj * denom_proj)) * (d_denom),
            torch.zeros_like(denom_proj),
        )
        # Final JVP: dx = dx_eq + dalpha * y + alpha * dy
        dx = dx_eq + dalpha.unsqueeze(-1) * y + alpha.unsqueeze(-1) * dy
        return dx


def find_start_point(wl, wh, L, U, p):
    denom = bdot(p, U).clamp_min(1e-30)
    x0 = torch.where(
        wl.unsqueeze(-1) < 0.0,
        torch.zeros_like(p),
        U * (wl / denom).unsqueeze(-1),
    )
    return x0


# @torch.compile
def solve_QP_problem(Q, m, c, wl, wh, L, U, p, x0):
    r = ActiveSetQPFunc.apply(Q, m, c, wl, wh, L, U, p, x0)
    return r[0]


# Call autograd.Function in eager mode
def solve_QP_problem2(Q, m, c, wl, wh, L, U, p, x0):
    return ActiveSetQPFunc.apply(Q, m, c, wl, wh, L, U, p, x0)[0]


def objective(Q: torch.Tensor, m: torch.Tensor, x: torch.Tensor):
    return bquad(Q, x) + bdot(m, x)


class StockSolver(torch.nn.Module):

    @staticmethod
    def _expect(name, t, shape):
        if tuple(t.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}; got {tuple(t.shape)}"
            )

    def __init__(
        self,
        Q: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        X: torch.Tensor,
        budget: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
    ):
        super().__init__()
        self.update(Q, m, c, X, budget, kappa, theta)

    def update(
        self,
        Q: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        X: torch.Tensor,
        budget: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
    ):
        self.register_buffer("Q", Q)
        if self.Q.ndim != 3:
            raise ValueError(
                f"Q must have shape (B, n, n); got {tuple(self.Q.shape)}"
            )
        B, n, n2 = self.Q.shape
        if n != n2:
            raise ValueError(
                f"Q must be square on its last two dims; got {n}x{n2}"
            )
        tensors = {
            "m": m,
            "c": c,
            "X": X,
            "budget": budget,
            "kappa": kappa,
            "theta": theta,
        }
        for name, t in tensors.items():
            self.register_buffer(name, t)

        self.register_buffer("B", torch.tensor(B))
        self.register_buffer("n", torch.tensor(n))

        # Validate input
        self._expect("m", m, (B, n))
        self._expect("X", X, (B, n))
        self._expect("c", c, (B,))
        self._expect("budget", budget, (self.B,))
        self._expect("kappa", kappa, (B,))
        self._expect("theta", theta, (B,))
        if not torch.all(budget > 0):
            raise ValueError("Each element of budget must be > 0")
        if not torch.all(kappa >= 0):
            raise ValueError("Each element of kappa must be >= 0")
        if not torch.all(theta > 0):
            raise ValueError("Each element of theta must be > 0")

    def _find_start_point(
        self,
        p: torch.Tensor,   # (B, n) 
        wl: torch.Tensor,  # (B,)
        wh: torch.Tensor,  # (B,)
        L: torch.Tensor,   # (B, n)
        U: torch.Tensor,   # (B, n)
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

        #assert p.shape == L.shape == U.shape, \
        #    f"Shapes must match: p {p.shape}, L {L.shape}, U {U.shape}"

        B, n = L.shape

        # Compute p^T L and p^T U (batchwise)
        p_dot_L = bdot(p, L)  # (B,)
        p_dot_U = bdot(p, U)  # (B,)
        den = p_dot_U - p_dot_L       # (B,)

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

    def _convert_price(self, p: torch.Tensor, x0: torch.Tensor):
        p0 = p.repeat(self.m.shape[0], 1)  # TODO: change to m.size(0)
        m1 = (
            -self.m
            + p
            + 2.0 * torch.bmm(self.X.unsqueeze(1), self.Q).squeeze(1)
        )
        L = -1.0 / p * (self.budget * self.kappa).unsqueeze(1) - self.X
        U = 1.0 / p * (self.budget * self.theta).unsqueeze(1) - self.X
        # because of sign, theta and kappa are swapped, this is not an error!
        # this is cash restriction !
        # replace self.theta, self.kappa to change cash budget multiplier
        wl = (self.budget * (1.0 - self.theta)) - (self.X * p0).sum(dim=1)
        wh = (self.budget * (1.0 + self.kappa)) - (self.X * p0).sum(dim=1)
        # validate starting point and update if needed
        with torch.no_grad():
            x0_dot_p = bdot(x0, p0)
            invalid = (
                torch.any(x0 < L)
                | torch.any(x0 > U)
                | torch.any(x0_dot_p > wh)
                | torch.any(x0_dot_p < wl)
            )
            x0_adj = torch.where(
                invalid, self._find_start_point(p0, wl, wh, L, U), x0
            )
        return p0, m1, wl, wh, L, U, x0_adj

    def forward(self, p: torch.Tensor):
        BTG = 1e30
        return self.solve_x0(p.to(self.Q.dtype), torch.full_like(self.X, BTG))

    def solve_x0(self, p: torch.Tensor, x0: torch.Tensor):
        p0, m1, wl, wh, L, U, x0_adj = self._convert_price(p, x0)
        return solve_QP_problem(self.Q, m1, self.c, wl, wh, L, U, p0, x0_adj)

    def solve_sum(self, p: torch.Tensor):
        return self.forward(p).sum(dim=0)

    def solve_sum_x0(self, p: torch.Tensor, x0: torch.Tensor):
        return self.solve_x0(p).sum(dim=0)

    def compute_primals(self, p: torch.Tensor):
        B = self.m.shape[0]
        n = p.shape[-1]

        # ---- Primal preprocessing (same as your _convert_price/forward) ----
        # we pass a dummy x0; it won't affect the result and dx0 = 0
        dummy_x0 = torch.zeros_like(self.X)

        p0 = p.repeat(B, 1)  # (B, n)

        # m1 = self.m - p + 2 * (X Q)
        m1 = (
            -self.m
            + p
            + 2.0 * torch.bmm(self.X.unsqueeze(1), self.Q).squeeze(1)
        )

        # L = -(budget*kappa)/p - X ;  U = (budget*theta)/p - X
        ak = (self.budget * self.kappa).unsqueeze(1)  # (B,1)
        at = (self.budget * self.theta).unsqueeze(1)  # (B,1)
        L = -ak / p - self.X
        U = at / p - self.X

        # wl = budget*(1-theta) - <X, p0>,  wh = budget*(1+kappa) - <X, p0>
        Xp = (self.X * p0).sum(dim=1)  # (B,)
        wl = (self.budget * (1.0 - self.theta)) - Xp
        wh = (self.budget * (1.0 + self.kappa)) - Xp

        # keep the same feasibility adjustment logic, but it won't matter for dx
        with torch.no_grad():
            x0_dot_p = (dummy_x0 * p0).sum(dim=1)
            invalid = (
                torch.any(dummy_x0 < L)
                | torch.any(dummy_x0 > U)
                | torch.any(x0_dot_p > wh)
                | torch.any(x0_dot_p < wl)
            )
            x0_adj = torch.where(
                invalid.unsqueeze(-1),
                self._find_start_point(p0, wl, wh, L, U),
                dummy_x0,
            )

        # ---- Primal call ----
        x = solve_QP_problem(self.Q, m1, self.c, wl, wh, L, U, p0, x0_adj)

        active_comp = (x == L) | (x == U)
        active_values = torch.where(
            active_comp, torch.where(x == L, L, U), 0.0
        )
        active_budget = (torch.abs(bdot(x, p) - wl) < 1e-9) | (
            torch.abs(bdot(x, p) - wh) < 1e-9
        )
        budget_w = torch.where(
            active_budget,
            torch.where(
                torch.abs(bdot(x, p) - wl) < 1e-9,
                wl,
                wh,
            ),
            0.0,
        )
        saved_vars = (
            self.Q,
            m1,
            wl,
            wh,
            L,
            U,
            p0,
            active_comp,
            active_values,
            active_budget,
            budget_w,
        )

        # ---- JVP of the QP solve (single linearized KKT solve) ----
        # This calls your closed-form JVP; no AD/tracing involved.
        primals = ActiveSetQPFunc._process_primals(saved_vars)
        return (x, p, ak, at, self.X) + primals

    @staticmethod
    def compute_tangent(dp: torch.Tensor, all_primals: tuple):
        # ---- Tangent preprocessing (differentiate everything above w.r.t. p) ----
        x, p, ak, at, X, *primals = all_primals

        B, n = x.shape

        # broadcast helpers
        dp0 = dp.unsqueeze(0).expand(B, -1)  # (B, n)
        invp2 = (p * p).unsqueeze(0)  # (1, n)

        # dm1/dp = -I  (broadcast over batches)
        dm = -dp0  # (B, n)

        # dL/dp =  (ak)/p^2 * dp   ;   dU/dp = -(at)/p^2 * dp
        dL = (ak / invp2) * dp0  # (B, n)
        dU = -(at / invp2) * dp0  # (B, n)

        # dwl/dp = dwh/dp = - sum_i X_{b,i} * dp_i
        dsum = (X * dp0).sum(dim=1)  # (B,)
        dwl = -dsum
        dwh = -dsum

        # dp0 is the input tangent for p0
        dp_in = dp0

        # constants (no dependence on p)
        dQ = torch.zeros((B, n, n), dtype=x.dtype, device=x.device)
        dc = torch.zeros((B, n), dtype=x.dtype, device=x.device)
        dx0 = torch.zeros((B, n), dtype=x.dtype, device=x.device)

        jv = ActiveSetQPFunc._jvp_int(
            tuple(primals), dQ, dm, dc, dwl, dwh, dL, dU, dp_in, dx0
        )
        return jv.sum(dim=0)

    @torch.no_grad()
    def linearize_forward(self, p: torch.Tensor):
        # Compute primals once
        self.all_primals = self.compute_primals(p)

    @torch.no_grad()
    def forward_jvp(self, dp: torch.Tensor):
        return StockSolver.compute_tangent(dp, self.all_primals)


class StockSolverParams(NamedTuple):
    Sigma: torch.tensor
    M: torch.tensor
    c: torch.tensor
    c: torch.Tensor
    X: torch.Tensor
    budget: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor


class StockSolverPrimals(NamedTuple):
    Sigma: torch.tensor
    M: torch.tensor
    X: torch.tensor
    wl: torch.tensor
    wh: torch.tensor
    L: torch.tensor
    U: torch.tensor
    p: torch.tensor
    active_comp: torch.tensor
    active_values: torch.tensor
    active_budget: torch.tensor
    budget_w: torch.tensor
    H: torch.tensor
    p_eff: torch.tensor
    Lc: torch.tensor
    x_eq: torch.tensor
    y: torch.tensor
    p_dot_xeq: torch.tensor
    denom_proj: torch.tensor
    alpha: torch.tensor
    x: torch.tensor
    ak: torch.tensor
    at: torch.tensor


class StockSolverFunc(torch.autograd.Function):
    @staticmethod
    def forward(params: StockSolverParams, p: torch.tensor, x0, torch_tensor):
        pass

    @staticmethod
    def linearize_forward(
        params: StockSolverParams, p: torch.Tensor
    ) -> StockSolverPrimals:
        pass

    @staticmethod
    def forward_jvp(
        primals: StockSolverPrimals, grad_p: torch.Tensor
    ) -> StockSolverPrimals:
        pass


class GMRESSolver:
    @staticmethod
    def _safe_normalize(x: torch.Tensor, thresh: float | None = None):
        """
        L2-normalize x; if ||x|| <= thresh, return (0 vector, 0).
        """
        if thresh is None:
            thresh = torch.finfo(x.dtype).eps

        norm = torch.linalg.vector_norm(x)  # scalar tensor
        thresh_t = torch.as_tensor(
            thresh, dtype=norm.dtype, device=norm.device
        )

        # Avoid div-by-zero by clamping the denominator
        safe_norm = torch.clamp(norm, min=thresh_t)

        y = x / safe_norm  # well-defined
        use = norm > thresh_t

        y = torch.where(use, y, torch.zeros_like(x))
        norm_out = torch.where(use, norm, norm.new_zeros(()))

        return y, norm_out

    @staticmethod
    def _iterative_classical_gram_schmidt(
        Q: torch.Tensor,
        x: torch.Tensor,
    ):
        """
        Classical GS with optional re-orth (\"twice is enough\").
        Returns (q_unit, r) where q_unit ⟂ cols(Q).
        """
        m, k = Q.shape
        assert x.shape == (m,)

        # First orthogonalization
        r0 = x @ Q
        q0 = x - (Q @ r0)  # (m,)
        # Second orthogonalization
        h1 = q0 @ Q
        r1 = r0 + h1
        q1 = q0 - (Q @ h1)
        # Final orthogonalization
        h = q1 @ Q
        r = r1 + h
        q = q1 - (Q @ h)
        return q, r

    def _kth_arnoldi_iteration(
        self,
        k: torch.Tensor,  # scalar int tensor
        V: torch.Tensor,  # (m, restart+1)
        H: torch.Tensor,  # (restart, restart+1)
    ):
        device, dtype = V.device, V.dtype
        m, ncols = V.shape
        restart = H.shape[0]  # ncols == restart+1

        # Select v_k without Python ints
        e_k = functional.one_hot(k.to(torch.int64), num_classes=ncols).to(
            dtype=dtype, device=device
        )  # (ncols,)
        v_k = V @ e_k  # (m,)

        # Arnoldi “apply, orthogonalize”
        w = self.forward_jvp(v_k)
        _, w_norm0 = GMRESSolver._safe_normalize(w)

        # Project against ALL columns in V (unused ones are zeros, so harmless)
        q, r = GMRESSolver._iterative_classical_gram_schmidt(
            V, w
        )  # r: (ncols,)

        tol = (
            torch.as_tensor(torch.finfo(dtype).eps, dtype=dtype, device=device)
            * w_norm0
        )
        unit_v, v_norm_1 = GMRESSolver._safe_normalize(q, thresh=tol)

        # Write V[:, k+1] with a one-hot mask (no Python indexing)
        e_kp1 = functional.one_hot(
            (k + 1).to(torch.int64), num_classes=ncols
        ).to(
            dtype=dtype, device=device
        )  # (ncols,)
        V_new = V * (1 - e_kp1).unsqueeze(0) + unit_v.unsqueeze(
            1
        ) * e_kp1.unsqueeze(
            0
        )  # (m, ncols)

        # Build H row k: [r_0..r_k, v_norm_1, 0..]
        # Set position k+1 to v_norm_1 using one-hot overwrite trick
        h_full = r + e_kp1 * (v_norm_1 - (r * e_kp1).sum())  # (ncols,)

        row_mask = (
            functional.one_hot(k.to(torch.int64), num_classes=restart)
            .to(dtype=dtype, device=device)
            .unsqueeze(1)
        )  # (restart,1)
        H_new = H * (1 - row_mask) + h_full.unsqueeze(0) * row_mask

        breakdown = v_norm_1 == 0
        return V_new, H_new, breakdown

    @staticmethod
    def loop_cond(V, H, breakdown, k, restart):
        return (k < restart) & (~breakdown)

    def arnoldi_process(self, V, H, breakdown, k, restart):
        V2, H2, breakdown2 = self._kth_arnoldi_iteration(k, V, H)
        return V2, H2, breakdown2, k + 1

    def _gmres_batched(
        self, b, x0, unit_residual, residual_norm, restart: int
    ):
        """
        One GMRES restart (left-preconditioned):
          - Builds up to `restart` Krylov vectors with Arnoldi using operator M @ A.
          - Solves min_y || H_j^T y - beta * e1 || and returns x = x0 + V_j y.
          - Returns (x, unit_residual_new, residual_norm_new).
        A: callable (m,) -> (m,)
        b, x0, unit_residual: (m,)
        residual_norm: scalar tensor
        M: (m, m) tensor (left preconditioner applied by @)
        """
        device, dtype = b.device, b.dtype
        m = b.shape[0]
        ncols = restart + 1
        # func_wrapper.set_function(f)

        # V: (m, restart+1); first column = unit_residual
        V0 = torch.zeros((m, ncols), dtype=dtype, device=device)
        V0[:, 0] = unit_residual

        # H: (restart, restart+1)
        H0 = torch.zeros((restart, ncols), dtype=dtype, device=device)

        # If no steps requested, skip the while_loop to avoid zero-iteration aliasing
        k0 = torch.tensor(0, dtype=torch.int64, device=device)
        br0 = torch.tensor(False, dtype=torch.bool, device=device)

        idx_cols = torch.arange(ncols, device=device)  # [0, 1, ..., restart]
        idx_rows = torch.arange(
            restart, device=device
        )  # [0, 1, ..., restart-1]
        eps = torch.as_tensor(
            torch.finfo(dtype).eps, dtype=dtype, device=device
        )

        V, H, _, k = ops.higher_order.while_loop(
            GMRESSolver.loop_cond,
            self.arnoldi_process,
            (V0, H0, br0, k0),
            (restart,),
        )
        # Solve least squares: (Hj^T) y ≈ beta * e1   with Hj = H[:j, :j+1]
        # Fixed-shape LS: solve min || H^T y - beta e1 ||
        # Build masked, fixed-size matrix A_ls (no Python slicing by k)
        ncols = H.shape[1]  # restart + 1
        idx_r = torch.arange(H.shape[0], device=H.device)  # 0..restart-1
        idx_c = torch.arange(ncols, device=H.device)  # 0..restart
        row_mask = (idx_c <= k).to(H.dtype)  # keep rows 0..k in H^T
        y_mask = (idx_r < k).to(H.dtype)  # unknowns y[0..k-1]

        Ht = H.transpose(0, 1)  # (ncols, restart)
        A_ls = Ht * y_mask.unsqueeze(0)  # zero inactive columns
        A_ls = A_ls * row_mask.unsqueeze(1)  # zero rows beyond k

        beta = torch.zeros(ncols, dtype=H.dtype, device=H.device)
        beta[0] = residual_norm.to(beta.dtype)

        # SVD-based LS: y_full = V * S^+ * U^T * beta
        U, S, VT = torch.linalg.svd(A_ls, full_matrices=False)
        tol = (
            torch.finfo(H.dtype).eps * max(A_ls.shape) * (S.max() + 1)
        )  # simple cutoff
        Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
        y_full = VT.transpose(0, 1) @ (
            Sinv * (U.transpose(0, 1) @ beta)
        )  # (restart,)

        dx = V[:, :-1] @ y_full
        x = x0 + dx

        # New (preconditioned) residual
        residual = self.compute_residual(b, x)

        unit_residual_new, residual_norm_new = GMRESSolver._safe_normalize(
            residual
        )

        return x, unit_residual_new, residual_norm_new

    @staticmethod
    def _gmres_cond_fun(x, k, ures, rnorm, b, maxiter, thresh, restart):
        return (k < maxiter) & (rnorm > thresh)

    def _gmres_body_fun(self, x, k, ures, rnorm, b, maxiter, thresh, restart):
        x_new, ures_new, rnorm_new = self._gmres_batched(
            b, x, ures, rnorm, restart
        )
        return x_new, k + 1, ures_new, rnorm_new

    def gmres(
        self,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        tol: float = 1e-5,
        atol: float = 0.0,
        restart: int = 20,
        maxiter: int | None = None,
    ):
        """
        GMRES solves A x = b.

        A: function taking a 1D tensor and returning a 1D tensor.
        Residual used for convergence is the *preconditioned* residual r = M @ (b - A(x)).
        Converged when ||r|| <= max(tol * ||b||, atol).

        A is specified as a function performing A(vi) -> vf = A @ vi, and in principle
        need not have any particular special properties, such as symmetry. However,
        convergence is often slow for nearly symmetric operators.

        Parameters
        ----------
        b : torch.tensor
            Right hand side of the linear system representing a single vector. Can be
            stored as an array or Python container of array(s) with any shape.

        Returns
        -------
        x : tensor
            The converged solution. Has the same structure as ``b``.

        Other Parameters
        ----------------
        x0 : tensor, optional
            Starting guess for the solution. Must have the same structure as ``b``.
            If this is unspecified, zeroes are used.
        tol, atol : float, optional
            Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
        restart : integer, optional
            Size of the Krylov subspace ("number of iterations") built between
            restarts. GMRES works by approximating the true solution x as its
            projection into a Krylov space of this dimension - this parameter
            therefore bounds the maximum accuracy achievable from any guess
            solution. Larger values increase both number of iterations and iteration
            cost, but may be necessary for convergence. The algorithm terminates
            early if convergence is achieved before the full subspace is built.
            Default is 20.
        maxiter : integer
            Maximum number of times to rebuild the size-``restart`` Krylov space
            starting from the solution found at the last iteration. If GMRES
            halts or is very slow, decreasing this parameter may help.
            Default is infinite.
        """

        assert b.ndim == 1, "This implementation expects 1D vectors."
        device, dtype = b.device, b.dtype
        m = b.shape[0]

        if x0 is None:
            x0 = torch.zeros_like(b)

        if maxiter is None:
            maxiter = 10 * m  # common default

        restart = int(min(restart, m))

        if x0.shape != b.shape:
            raise ValueError("x0 and b must have matching shape")

        # Build tensor tolerances
        b_norm = torch.linalg.vector_norm(b)
        atol_t = torch.as_tensor(atol, dtype=dtype, device=device)
        thresh = torch.maximum(
            torch.as_tensor(tol, dtype=dtype, device=device) * b_norm, atol_t
        )

        # Initial (left-preconditioned) residual
        residual0 = self.compute_residual(b, x0)
        unit_residual, residual_norm = GMRESSolver._safe_normalize(residual0)

        # Early exit if already converged
        if residual_norm <= thresh:
            return x0

        # while_loop state must be tensors
        k0 = torch.tensor(0, dtype=torch.int64, device=device)

        (
            x_final,
            *_,
        ) = ops.higher_order.while_loop(
            GMRESSolver._gmres_cond_fun,
            self._gmres_body_fun,
            (x0, k0, unit_residual, residual_norm),
            (b, torch.as_tensor(maxiter), torch.as_tensor(thresh), restart),
        )

        return x_final


def newton_krylov(
    s,
    x0,
    S,
    tol=1e-8,
    rtol=1e-8,
    lstol=1e-6,
    max_outer=50,
    gmres_restart=20,
    gmres_maxiter=200,
    positive=False,
    verbose=False,
):
    """
    Solve f(x)=0 with Newton–Krylov (GMRES) using only JVPs.

    f : callable x -> F(x) (same shape),
    autograd-friendly (use torch.stack, not torch.tensor([...])).
    x0 : 1D tensor initial guess.

    Returns x
    """
    trajectory = []
    x = x0.clone().detach()

    F = s.solve_sum(x) - S
    norm0 = torch.linalg.vector_norm(F)
    if norm0 == 0.0:
        return x

    for it in range(1, max_outer + 1):
        r = F.clone()

        # Inexact Newton forcing term (Eisenstat–Walker-like)
        eta = min(
            0.1, 0.5 * math.sqrt(max(1e-300, torch.linalg.norm(F) / norm0))
        )

        # Linear operator for GMRES
        s.linearize_forward(x)

        # Solve J(x) Δ = -F(x)
        dx = s.gmres(
            -r,
            x0=x.clone(),
            tol=eta,
            atol=0.0,
            restart=gmres_restart,
            maxiter=gmres_maxiter,
        )
        F0 = torch.linalg.vector_norm(F)
        t = 1.0
        while t > lstol:
            x_trial = x + t * dx
            if torch.all(x_trial > 0.0) or not positive:
                F_trial = s.solve_sum(x_trial) - S
                if torch.linalg.vector_norm(F_trial) <= F0 and (
                    (
                        torch.all(x_trial > 0)
                        and torch.all(x_trial < 2 * x)
                        and torch.all(x_trial > x / 2)
                    )
                    or not positive
                ):
                    break
            t *= 0.5
        if t >= lstol:
            x, F = x_trial.detach(), F_trial.detach()
        res = torch.linalg.norm(F)
        if verbose:
            step_size = float(torch.norm(t * dx).item())
            # print(x)
            print(
                f"[NK] it={it:02d}  ||F||={res:.4e}  step={t:.4e} "
                f"step_size = {step_size:.4e}"
            )
        if res <= tol or res <= rtol * norm0 or t <= lstol:
            return x

    return x


class StockSolverGMRES(StockSolver, GMRESSolver):
    def compute_residual(self, b, x):
        return b - self.forward_jvp(x)


class BalanceFunc(GMRESSolver):
    def __init__(self):
        pass

    def _save(self, Q, m, c, X, budget, kappa, theta, S, p_star):
        self.Q = Q
        self.m = m
        self.c = c
        self.X = X
        self.budget = budget
        self.kappa = kappa
        self.theta = theta
        self.S = S
        self.p_star = p_star

    def update(self, Q, m, c, X, budget, kappa, theta, S, p_star):
        self._save(Q, m, c, X, budget, kappa, theta, S, p_star)
        stock_solver.update(Q, m, c, X, budget, kappa, theta)
        # (∂F/∂p) u  at the fixed point p_star
        stock_solver.linearize_forward(p_star)

    def forward_jvp(self, dp):
        return stock_solver.forward_jvp(dp)

    def compute_residual(self, b, x):
        return b - stock_solver.forward_jvp(x)


stock_solver = StockSolverGMRES(
    torch.empty(1, 1, 1),
    torch.empty(1, 1),
    torch.empty(1),
    torch.zeros(1, 1),
    torch.ones(1),
    torch.zeros(1),
    torch.ones(1),
)


def stock_solver_func(Q, m, c, X, budget, kappa, theta, S, x0, p):
    stock_solver.update(Q, m, c, X, budget, kappa, theta)
    return stock_solver.solve_sum(p) - S


balance_func = BalanceFunc()


def balance_func_gmres(
    Q, m, c, X, budget, kappa, theta, S, x0, p_star, grad_p
):
    balance_func.update(Q, m, c, X, budget, kappa, theta, S, p_star)
    return balance_func.gmres(
        grad_p,
        x0=x0,
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=50,
    )


class PriceSolver(torch.autograd.Function):
    @staticmethod
    def forward(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        c: torch.Tensor,  # (B, )
        X: torch.Tensor,  # (B, n)
        budget: torch.Tensor,  # (B, )
        kappa: torch.Tensor,  # (B, )
        theta: torch.Tensor,  # (B, )
        S: torch.Tensor,  # (n, )
        p0=None,
    ):
        if p0 is None:
            p0 = (m * budget.unsqueeze(-1)).sum(dim=0) / (
                budget.sum(dim=0) + 1e-4
            )
        stock_solver.update(Q, m, c, X, budget, kappa, theta)
        return newton_krylov(
            stock_solver,
            p0,
            S,
            tol=1e-8,
            rtol=1e-8,
            lstol=1e-6,
            gmres_restart=10,
            gmres_maxiter=10,
            max_outer=40,
            positive=True,
            verbose=False,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        Q, m, c, X, budget, kappa, theta, S, x0 = inputs
        if x0 is None:
            x0 = (m * budget.unsqueeze(-1)).sum(dim=0) / (
                budget.sum(dim=0) + 1e-4
            )

        p_star = output

        # Save for backward (VJP path)
        ctx.save_for_backward(
            Q,
            m,
            c,
            X,
            budget,
            kappa,
            theta,
            S,
            x0,
            p_star.detach().requires_grad_(True),
        )

    @staticmethod
    def backward(ctx, grad_p):
        Q, m, c, X, budget, kappa, theta, S, x0, p_star = ctx.saved_tensors

        # Solve A v = grad_p with GMRES
        v = balance_func_gmres(*ctx.saved_tensors, grad_p)

        with torch.enable_grad():
            F = stock_solver_func(*ctx.saved_tensors)

        input_tuple = tuple(
            ctx.saved_tensors[i]
            for i in range(len(ctx.needs_input_grad))
            if ctx.needs_input_grad[i]
        )
        # Compute
        grad_tuple = torch.autograd.grad(
            F,
            input_tuple,
            grad_outputs=-v,
            retain_graph=False,
            allow_unused=True,
        )
        gidx = torch.cumsum(torch.tensor(ctx.needs_input_grad), dim=0) - 1
        return tuple(
            grad_tuple[gidx[i]] if ctx.needs_input_grad[i] else None
            for i in range(9)
        )


def compute_initial_prices(Sigma, M, budget, kappa, theta, S, p0=None):
    return PriceSolver.apply(
        Sigma,
        M,
        torch.zeros_like(budget),
        torch.zeros_like(M),
        budget,
        kappa,
        theta,
        S,
        p0,
    )


def find_equilibrium_prices(
    Sigma,
    expected_returns,
    commission,
    holdings,
    budget,
    short_leverage,
    long_leverage,
    supply,
    initial_approximation=None,
):
    """
    Compute market-clearing (equilibrium) prices for a *batch* of heterogeneous-agent markets.

    This function **only accepts batched inputs** (single instances are disallowed).
    All parameters must be stacked along dimension 0 (batch). Outputs are (B, N).

    Accepts PyTorch tensors or array-likes convertible to tensors and fully supports autograd,
    enabling gradient-based calibration and inference (e.g., MCMC/NUTS).

    Parameters
    ----------
    Sigma : (B, N, N)
        Expected covariance matrices (PSD) per batch item.
    expected_returns : (B, N)
        Expected returns per asset per batch item.
    commission : (B, N)
        Per-asset commission to **buy**. Use zeros for initial portfolios.
    holdings : (B, N)
        Current holdings per batch item. Use zeros for initial portfolios.
    budget : (B,)
        Total budget per batch item.
    short_leverage : (B,) or (B, N)
        Per-security short cap as a fraction of `budget`. `0` ⇒ no short sales.
    long_leverage : (B,) or (B, N)
        Per-security long cap as a fraction of `budget`. Values < 1 ⇒ no leverage.
    supply : (B, N)
        Exogenous supply of each security per batch item.
    initial_approximation : (B, N), optional
        **Strictly positive** initial guess for prices. It should not affect the final equilibrium,
        but a good guess can improve performance when close to equilibrium.

    Returns
    -------
    equilibrium_prices : (B, N) torch.Tensor
        Market-clearing price vectors (aggregate demand ≈ supply per batch).

    Notes
    -----
    - **Batched only**: all inputs (including `initial_approximation`, if provided) must be batched.
    - Fully differentiable; non-tensors are converted to tensors.

    Examples
    --------
    >>> p_eq = find_equilibrium_prices(
    ...     Sigma_b, mu_b, commission_b, holdings_b, budget_b,
    ...     short_leverage_b, long_leverage_b, supply_b,
    ...     initial_approximation=torch.ones_like(supply_b)
    ... )
    """    
    return PriceSolver.apply(
        torch.as_tensor(Sigma, dtype=torch.float64),
        torch.as_tensor(expected_returns, dtype=torch.float64),
        torch.as_tensor(commission, dtype=torch.float64),
        torch.as_tensor(holdings, dtype=torch.float64),
        torch.as_tensor(budget, dtype=torch.float64),
        torch.as_tensor(short_leverage, dtype=torch.float64),
        torch.as_tensor(long_leverage, dtype=torch.float64),
        torch.as_tensor(supply, dtype=torch.float64),
        (
            None
            if initial_approximation is None
            else torch.as_tensor(initial_approximation, dtype=torch.float64)
        ),
    )

def optimize_portfolio(
    Sigma,
    expected_returns,
    commission,
    holdings,
    budget,
    short_leverage,
    long_leverage,
    prices,
):
    """
    Optimize a single portfolio or a batch of portfolios under budget, short, and long leverage constraints.

    Batching
    --------
    Supports single or batched optimization. When batched, **all parameters except `prices` must be
    stacked along dimension 0** (the batch dimension). **`prices` is a market-wide vector shared by
    everyone and MUST be unbatched** (shape (N,)); it will be internally broadcast across the batch.

    The method accepts PyTorch tensors or any array-like objects convertible to `torch.Tensor`,
    and is fully differentiable (autograd-friendly).

    Parameters
    ----------
    Sigma : (N, N) or (B, N, N)
        Expected covariance matrix (PSD). Batched as (B, N, N).
    expected_returns : (N,) or (B, N)
        Expected returns. Batched as (B, N).
    commission : (N,) or (B, N)
        Per-asset commission to **buy**. Use zeros for initial portfolios.
    holdings : (N,) or (B, N)
        Current holdings (shares). Use zeros for initial portfolios.
    budget : () or (B,)
        Total budget (scalar or batched scalar).
    short_leverage : () or (N,) or (B,) or (B, N)
        Per-security short cap as a fraction of `budget`. `0` ⇒ no short sales.
    long_leverage : () or (N,) or (B,) or (B, N)
        Per-security long cap as a fraction of `budget`. Values < 1 ⇒ no leverage.
    prices : (N,)
        **Market prices** shared by all agents. **Must be 1-D (N,)**; batches are not allowed.

    Returns
    -------
    optimal_holdings : (N,) or (B, N) torch.Tensor
        Optimal post-trade holdings satisfying constraints.

    Notes
    -----
    - `prices` is **unbatched** by design (market-wide).
    - All other inputs follow the single vs. batched rules above.
    - Fully differentiable; non-tensors are converted to tensors.

    Examples
    --------
    Single portfolio
    >>> h = optimize_portfolio(Sigma, mu, commission=torch.zeros_like(mu),
    ...     holdings=torch.zeros_like(mu), budget=1.0,
    ...     short_leverage=0.0, long_leverage=1.0, prices=prices)

    Batched portfolios (B, N) with shared market prices (N,)
    >>> h_b = optimize_portfolio(Sigma_b, mu_b, commission_b, holdings_b,
    ...     budget_b, short_leverage=0.0, long_leverage=1.0, prices=prices)
    """    
    if Sigma.ndim == 2:
        ssolv = StockSolver(
            torch.as_tensor(Sigma, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(expected_returns, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(commission, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(holdings, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(budget, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(short_leverage, dtype=torch.float64).unsqueeze(0),
            torch.as_tensor(long_leverage, dtype=torch.float64).unsqueeze(0),
        )        
    else:
        ssolv = StockSolver(
            torch.as_tensor(Sigma, dtype=torch.float64),
            torch.as_tensor(expected_returns, dtype=torch.float64),
            torch.as_tensor(commission, dtype=torch.float64),
            torch.as_tensor(holdings, dtype=torch.float64),
            torch.as_tensor(budget, dtype=torch.float64),
            torch.as_tensor(short_leverage, dtype=torch.float64),
            torch.as_tensor(long_leverage, dtype=torch.float64),
        )
    return ssolv(torch.as_tensor(prices, dtype=torch.float64))
