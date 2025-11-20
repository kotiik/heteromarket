import torch
from torch import ops
import torch.nn.functional as functional
from typing import Any, Tuple, Sequence, NamedTuple, Callable, Tuple


def _as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


class ExplicitADFunction(torch.autograd.Function):
    """
    Base class for stateless, formula-based autograd with primals-only-once,
    modern-style setup_context, optional non-differentiable outputs, and
    needs_input_grad-aware backward.

    Subclass MUST implement:
      - compute(*inputs) -> outputs
      - compute_primals(*inputs, outputs) -> saved
      - vjp_from_primals(saved, *cotangents, needs_input_grad=None) -> grads_per_input
      - jvp_from_primals(saved, *tangents) -> out_tangents

    Optional in subclass:
      - non_differentiable_output_indices: tuple[int, ...] (class attribute)
    """

    # Set in subclasses as needed, e.g., (1, 3)
    non_differentiable_output_indices: Tuple[int, ...] = ()

    # --------- hooks subclasses must provide ----------
    @staticmethod
    def compute(*inputs: torch.Tensor) -> Tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Compute the forward outputs of the function from the given inputs.

        This method defines the mathematical transformation implemented by the
        custom operation. It is called once during the forward pass (both in
        eager execution and compiled graphs) and must return the function’s
        outputs as Tensors.

        The computation performed here is pure and stateless: given identical
        inputs, it must always produce identical outputs. No autograd context is
        active during this call, and the returned tensors form the “outputs” of
        the forward pass. Any additional intermediate data required to compute
        derivatives should instead be prepared in :meth:`compute_primals`, which
        is executed later inside :meth:`setup_context`.

        Parameters
        ----------
        *inputs : Tensor
            The input tensors of the operation. Each may or may not require
            gradients. These inputs correspond exactly to the arguments passed
            to :meth:`apply`.

        Returns
        -------
        outputs : Tensor or tuple[Tensor, ...]
            The result of applying the function to the given inputs. This can be
            a single Tensor or a tuple of Tensors. The structure and shapes must
            match what is expected by subsequent consumers of the operation.

        Notes
        -----
        - This method must **not** perform any in-place modifications on its
          inputs.
        - The returned tensors should be free of autograd history; any
          intermediate values needed for differentiation should be recomputed or
          cached in :meth:`compute_primals`.
        - Subclasses should aim for numerical stability and efficiency in this
          computation, as it is executed during both training and inference.
        """
        raise NotImplementedError

    @staticmethod
    def compute_primals(
        *inputs: torch.Tensor, outputs: Tuple[torch.Tensor, ...] | torch.Tensor
    ) -> Any:
        """
        Prepare and return any auxiliary data (“primals”) required to compute
        derivatives efficiently.

        This method is executed once inside :meth:`setup_context`, after the
        forward outputs have already been computed by :meth:`compute`. Its sole
        purpose is to precompute and cache any intermediate quantities needed
        later by :meth:`vjp_from_primals` or :meth:`jvp_from_primals`, so that
        these derivatives can be evaluated without recomputing the entire
        forward function.

        Implementations should return an arbitrary Python object (typically a
        tuple or dictionary of tensors, scalars, or constants) that will be
        stored on the autograd context as ``ctx._saved``. This data must not
        depend on autograd history or contain tensors that keep the computation
        graph alive.

        Parameters
        ----------
        *inputs : Tensor
            The same input tensors originally passed to :meth:`compute`. These
            are provided so that intermediate quantities depending on the inputs
            can be precomputed.

        outputs : Tensor or tuple[Tensor, ...]
            The outputs returned by :meth:`compute`. This allows derivatives to
            be expressed in terms of both inputs and outputs if desired.

        Returns
        -------
        saved : Any
            Arbitrary auxiliary data required for derivative computation.
            Common examples include reusable partial results, constants,
            or values of intermediate expressions. The returned object is passed
            verbatim to both :meth:`vjp_from_primals` and :meth:`jvp_from_primals`
            during backward and forward-mode differentiation, respectively.

        Notes
        -----
        - This method should be **pure** and side-effect-free.
        - It must not perform in-place operations on its inputs or outputs.
        - The returned data should be as lightweight as possible — avoid storing
          entire input tensors if only small derived quantities are needed.
        - This method is called exactly once per forward pass.
        """
        raise NotImplementedError

    @staticmethod
    def vjp_from_primals(
        saved: Any,
        *cotangents: torch.Tensor,
        needs_input_grad: Sequence[bool] | None = None,
    ) -> Tuple[torch.Tensor | None, ...]:
        """
        Compute the vector–Jacobian product (VJP) of the function with respect to
        its inputs, given precomputed primal data.

        This method implements the **reverse-mode derivative** of the function,
        i.e. the mapping:
            (cotangents) ↦ (gradients w.r.t. inputs)

        It is invoked during :meth:`backward` and in functional VJP evaluations.
        Unlike `compute_primals`, it must not build any autograd graph — the
        returned gradients must be computed manually using closed-form formulas.

        Parameters
        ----------
        saved : Any
            The auxiliary data returned by :meth:`compute_primals`. Typically
            includes any intermediate tensors or constants needed to evaluate
            partial derivatives efficiently. It must not reference autograd
            history (no tensors with `grad_fn` attached).

        *cotangents : Tensor
            One tensor per differentiable output of the function, representing
            the adjoint (or "upstream gradient") associated with that output.
            Each has the same shape as its corresponding output tensor.

        needs_input_grad : Sequence[bool] or None, optional
            A boolean mask indicating which inputs actually require gradients,
            as provided by `ctx.needs_input_grad` during the backward pass.
            Implementations should use this to skip unnecessary computation and
            may return `None` for inputs that do not require gradients. If
            `None`, gradients for all inputs are assumed to be needed.

        Returns
        -------
        gradients : tuple[Tensor or None, ...]
            One element per input tensor to the forward function. Each element
            is either:
              - a `Tensor` containing the gradient of the scalar objective
                with respect to that input, or
              - `None`, if no gradient is required for that input.

        Notes
        -----
        - The shapes of the returned gradients must match the corresponding
          input tensors.
        - This method should be **pure** (no side effects) and must not rely on
          any autograd computation.
        - For non-differentiable outputs (listed in
          `non_differentiable_output_indices`), the corresponding cotangents may
          be ignored.
        - To avoid unnecessary allocations, implementations may compute
          gradients in-place on temporary buffers, but must not modify
          user-provided inputs or saved data.
        """
        raise NotImplementedError

    @staticmethod
    def jvp_from_primals(
        saved: Any, *tangents: torch.Tensor | None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the Jacobian–vector product (JVP) of the function with respect to
        its inputs, given precomputed primal data.

        This method implements the **forward-mode derivative** of the function,
        i.e. the mapping:
            (tangents) ↦ (output tangents)

        It is invoked during forward-mode automatic differentiation (e.g.
        :meth:`jvp`) and in functional JVP evaluations. The implementation must
        use explicit derivative formulas rather than relying on PyTorch’s
        autograd tracing.

        Parameters
        ----------
        saved : Any
            The auxiliary data returned by :meth:`compute_primals`. It contains
            any intermediate tensors or constants needed to compute derivatives.
            This data must not depend on autograd state or contain tensors with
            active computation graphs.

        *tangents : Tensor or None
            One tangent vector per input tensor. Each tangent represents the
            infinitesimal change in that input. If a tangent is `None`, it should
            be treated as a zero tensor of the same shape as the corresponding
            input. Implementations are responsible for substituting zeros when
            needed.

        Returns
        -------
        output_tangents : tuple[Tensor, ...]
            The tangent vectors corresponding to the function’s outputs. Each
            element must have the same shape as the respective output tensor from
            :meth:`compute_primals`. Non-differentiable outputs (listed in
            `non_differentiable_output_indices`) should yield zero tangents.

        Notes
        -----
        - This method provides **forward-mode differentiation**, complementing
          the reverse-mode derivative defined in :meth:`vjp_from_primals`.
        - Implementations must not rely on PyTorch’s autograd engine.
        - Tangent computation should be numerically stable and efficient;
          precomputing reusable quantities in :meth:`compute_primals` is
          encouraged.
        - The function should be **pure**: no in-place modification of inputs or
          saved data is allowed.
        - All output tangents must be returned as `Tensor` objects; use zeros for
          non-differentiable or unused outputs rather than returning `None`.
        """
        raise NotImplementedError

    # --------------- autograd plumbing ----------------
    @classmethod
    def forward(cls, *inputs: torch.Tensor):
        return cls.compute(*inputs)

    @classmethod
    def setup_context(cls, ctx, inputs, output):
        # 1) Mark non-differentiable outputs (if any)
        nd_idx = getattr(cls, "non_differentiable_output_indices", ())
        if nd_idx:
            outs = _as_tuple(output)
            to_mark = [outs[i] for i in nd_idx if i < len(outs)]
            if to_mark:
                ctx.mark_non_differentiable(*to_mark)

        # 2) Compute and stash primals/intermediates once
        saved = cls.compute_primals(*inputs, outputs=output)
        ctx._saved = saved

    @classmethod
    def backward(cls, ctx, *cotangents: torch.Tensor):
        needs = ctx.needs_input_grad  # tuple[bool] matching forward inputs
        grads = list(
            cls.vjp_from_primals(
                ctx._saved, *cotangents, needs_input_grad=needs
            )
        )
        # Ensure None for inputs that don't need grad (safety + perf)
        for i, need in enumerate(needs):
            if not need:
                grads[i] = None
        return tuple(grads)

    @classmethod
    def jvp(cls, ctx, *tangents: torch.Tensor | None):
        # PyTorch may pass None tangents; subclass should treat them as zeros
        return cls.jvp_from_primals(ctx._saved, *tangents)

    # --------------- functional helpers ----------------
    @classmethod
    def fwd(cls, *inputs: torch.Tensor):
        with torch.no_grad():
            outputs, _ = cls.compute_primals(*inputs)
        return outputs

    @classmethod
    def primals(cls, *inputs: torch.Tensor):
        outputs, saved = cls.compute_primals(*inputs)
        return outputs, saved

    @classmethod
    def vjp_with_primals(
        cls, saved: Any, *cotangents: torch.Tensor, needs_input_grad=None
    ):
        return cls.vjp_from_primals(
            saved, *cotangents, needs_input_grad=needs_input_grad
        )

    @classmethod
    def jvp_with_primals(cls, saved: Any, *tangents: torch.Tensor):
        return cls.jvp_from_primals(saved, *tangents)


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


class SolverParameters(NamedTuple):
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


class SolverPrimalsComplete(NamedTuple):
    """
    Complete set of values for backpropogation of gradient
    """

    Q: torch.Tensor  # (B, n, n)
    m: torch.Tensor  # (B, n)
    wl: torch.Tensor  # (B, )
    wh: torch.Tensor  # (B, )
    L: torch.Tensor  # (B, n)
    U: torch.Tensor  # (B, n)
    p: torch.Tensor  # (B, n)
    active_comp: torch.Tensor  # (B, n), boolean
    active_values: torch.Tensor  # (B, n)
    active_budget: torch.Tensor  # (B, ), boolean
    budget_w: torch.Tensor  # (B, )
    H: torch.Tensor  # (B, n, n)
    p_eff: torch.Tensor
    Lc: torch.Tensor
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


class ActiveSetQPFunc(ExplicitADFunction):
    non_differentiable_output_indices = (1, 2, 3, 4)

    @staticmethod
    def _unpack_state_params(args: tuple):
        n_state = len(SolverState._fields)
        return SolverState(*args[:n_state]), SolverParameters(*args[n_state:])

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
            mask | (alpha_budget_low < -eps) | (p_dot_delta > 0.0),
            1.0,
            alpha_budget_low,
        )
        alpha_budget_high = torch.where(
            mask | (alpha_budget_high < -eps) | (p_dot_delta < 0.0),
            1.0,
            alpha_budget_high,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:  # g: (B, n)  # g_along_p: (B,)
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
        torch.Tensor,  # low_keep (B, n)
        torch.Tensor,  # high_keep (B, n)
        torch.Tensor,  # zero_keep (B, n)
        torch.Tensor,  # budg_low_keep (B, )
        torch.Tensor,  # budg_high_keep  (B, )
    ]:
        """
        Update active-set masks based on KKT sign rules at the new point.
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
        state: SolverState, params: SolverParameters
    ) -> tuple[
        torch.Tensor,  # active_comp (B, n)
        torch.Tensor,  # active_values (B, n)
        torch.Tensor,  # active_budget (B, )
        torch.Tensor,  # budget_w  (B, )
    ]:
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
    def _exit_cond(*args):
        state, _ = ActiveSetQPFunc._unpack_state_params(args)
        return (~state.done).any()

    # takes SolverState and SolverParams in unpacked form
    @staticmethod
    def _main_loop(*args):
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
        active_comp = (x <= L + eps) | (x >= U - eps)
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

    # ------------ Interface methods ------------------
    @staticmethod
    def compute(
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
            _active_set_exit_cond,
            _active_set_main_loop,
            tuple(state),
            (Q, m, c, wl, wh, L, U, p),
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
    def compute_primals(
        *inputs: torch.Tensor, outputs: Tuple[torch.Tensor, ...] | torch.Tensor
    ) -> SolverPrimalsComplete:
        Q, m, _, wl, wh, L, U, p, _ = inputs
        _, active_comp, active_values, active_budget, budget_w = outputs

        primals = ActiveSetQPFunc._solve_under_active_int(
            Q, m, p, active_comp, active_values, active_budget, budget_w
        )
        return SolverPrimalsComplete(
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
    def vjp_from_primals(
        saved: SolverPrimalsComplete,
        *cotangents: torch.Tensor,
        needs_input_grad: Sequence[bool] | None = None,
    ) -> Tuple[torch.Tensor | None, ...]:
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
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = saved
        grad_x, _, _, _, _ = cotangents
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
            grad_Q,
            grad_m,
            None,
            grad_wl,
            grad_wh,
            grad_L,
            grad_U,
            grad_p,
            None,
        )

    @staticmethod
    def jvp_from_primals(
        saved: SolverPrimalsComplete, *tangents: torch.Tensor | None
    ) -> Tuple[torch.Tensor, ...]:
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
        ) = saved
        dQ, dm, dc, dwl, dwh, dL, dU, dp, dx0 = tangents
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
        return (dx, None, None, None, None)


# Wrappers that Dynamo sees as plain functions
def _active_set_exit_cond(*state_and_params):
    return ActiveSetQPFunc._exit_cond(*state_and_params)
 

def _active_set_main_loop(*state_and_params):
    return ActiveSetQPFunc._main_loop(*state_and_params)


class StockSolverPrimals(NamedTuple):
    x: torch.Tensor
    p: torch.Tensor
    budget: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor
    ak: torch.Tensor
    at: torch.Tensor
    X: torch.Tensor
    Q: torch.Tensor


# TODO: Q and (maybe) p can be taken form QP solver primals


class StockSolverFunc(ExplicitADFunction):
    non_differentiable_output_indices = (2, 8)

    @staticmethod
    def _unpack_primals(args: tuple):
        n_state = len(StockSolverPrimals._fields)
        return StockSolverPrimals(*args[:n_state]), SolverPrimalsComplete(
            *args[n_state:]
        )

    @staticmethod
    def _expect(name, t, shape):
        if tuple(t.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}; got {tuple(t.shape)}"
            )

    @staticmethod
    def _validate(
        Q: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        X: torch.Tensor,
        budget: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
    ):
        if Q.ndim != 3:
            raise ValueError(
                f"Q must have shape (B, n, n); got {tuple(Q.shape)}"
            )
        B, n, n2 = Q.shape
        if n != n2:
            raise ValueError(
                f"Q must be square on its last two dims; got {n}x{n2}"
            )
        # Validate input
        StockSolverFunc._expect("m", m, (B, n))
        StockSolverFunc._expect("X", X, (B, n))
        StockSolverFunc._expect("c", c, (B,))
        StockSolverFunc._expect("budget", budget, (B,))
        StockSolverFunc._expect("kappa", kappa, (B,))
        StockSolverFunc._expect("theta", theta, (B,))
        if not torch.all(budget > 0):
            raise ValueError("Each element of budget must be > 0")
        if not torch.all(kappa >= 0):
            raise ValueError("Each element of kappa must be >= 0")
        if not torch.all(theta > 0):
            raise ValueError("Each element of theta must be > 0")

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

    @staticmethod
    def _convert_price(
        Q: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        X: torch.Tensor,
        budget: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        p: torch.Tensor,
        x0: torch.Tensor,
    ):
        p = p.to(dtype=Q.dtype, device=Q.device)
        m = m.to(dtype=Q.dtype, device=Q.device)
        X = X.to(dtype=Q.dtype, device=Q.device)
        p0 = p.repeat(m.shape[0], 1)  # TODO: change to m.size(0)
        m1 = -m + p + 2.0 * torch.bmm(X.unsqueeze(1), Q).squeeze(1)
        L = -1.0 / p * (budget * kappa).unsqueeze(1) - X
        U = 1.0 / p * (budget * theta).unsqueeze(1) - X
        # because of sign, theta and kappa are swapped, this is not an error!
        # this is cash restriction !
        # replace self.theta, self.kappa to change cash budget multiplier
        wl = (budget * (1.0 - theta)) - (X * p0).sum(dim=1)
        wh = (budget * (1.0 + kappa)) - (X * p0).sum(dim=1)
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
                invalid,
                StockSolverFunc._find_start_point(p0, wl, wh, L, U),
                x0,
            )
        return p0, m1, wl, wh, L, U, x0_adj

    # --------- hooks subclasses must provide ----------
    @staticmethod
    def compute(
        Q: torch.Tensor,  # (B, n, n)
        m: torch.Tensor,  # (B, n)
        c: torch.Tensor,  # (B, )
        X: torch.Tensor,  # (B, n)
        budget: torch.Tensor,  # (B, )
        kappa: torch.Tensor,  # (B, )
        theta: torch.Tensor,  # (B, )
        p: torch.Tensor,  # (n)
        x0: torch.Tensor,  # (B, n)
    ) -> Tuple[torch.Tensor, ...] | torch.Tensor:
        StockSolverFunc._validate(Q, m, c, X, budget, kappa, theta)
        p0, m1, wl, wh, L, U, x0_adj = StockSolverFunc._convert_price(
            Q, m, c, X, budget, kappa, theta, p, x0
        )
        return ActiveSetQPFunc.compute(Q, m1, c, wl, wh, L, U, p0, x0_adj)[0]

    @staticmethod
    def compute_primals(
        *inputs: torch.Tensor, outputs: Tuple[torch.Tensor, ...] | torch.Tensor
    ) -> Any:
        Q, m, c, X, budget, kappa, theta, p, _ = inputs

        B = m.shape[0]
        n = p.shape[-1]

        # ---- Primal preprocessing (same as your _convert_price/forward) ----
        # we pass a dummy x0; it won't affect the result and dx0 = 0
        dummy_x0 = torch.zeros_like(X)

        p0 = p.repeat(B, 1)  # (B, n)

        # m1 = m - p + 2 * (X Q)
        m1 = -m + p + 2.0 * torch.bmm(X.unsqueeze(1), Q).squeeze(1)

        # L = -(budget*kappa)/p - X ;  U = (budget*theta)/p - X
        ak = (budget * kappa).unsqueeze(1)  # (B,1)
        at = (budget * theta).unsqueeze(1)  # (B,1)
        L = -ak / p - X
        U = at / p - X

        # wl = budget*(1-theta) - <X, p0>,  wh = budget*(1+kappa) - <X, p0>
        Xp = (X * p0).sum(dim=1)  # (B,)
        wl = (budget * (1.0 - theta)) - Xp
        wh = (budget * (1.0 + kappa)) - Xp

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
                StockSolverFunc._find_start_point(p0, wl, wh, L, U),
                dummy_x0,
            )

        # ---- Primal call ----
        outputs = ActiveSetQPFunc.apply(Q, m1, c, wl, wh, L, U, p0, x0_adj)
        x = outputs[0]

        # ---- JVP of the QP solve (single linearized KKT solve) ----
        # This calls your closed-form JVP; no AD/tracing involved.
        inputs = Q, m1, None, wl, wh, L, U, p0, None
        # outputs = None, active_comp, active_values, active_budget, budget_w
        primals = ActiveSetQPFunc.compute_primals(*inputs, outputs=outputs)
        return (
            StockSolverPrimals(
                outputs[0], p, budget, kappa, theta, ak, at, X, Q.clone() # To avoid aliasing
            )
            + primals
        )

    @staticmethod
    def vjp_from_primals(
        saved: Any,  # saved primals
        *cotangents: torch.Tensor,
        needs_input_grad: Sequence[bool] | None = None,
    ) -> Tuple[torch.Tensor | None, ...]:
        """
        VJP to original inputs using ActiveSetQPFunc.vjp_from_primals.
        Returns grads in order:
          (gQ, gm, gc, gX, gbudget, gkappa, gtheta, gp, gx0)
        """
        # Single differentiable output (x); rest are non-diff
        (v, *_) = cotangents
        primals, inner_saved = StockSolverFunc._unpack_primals(saved)
        x, p, budget, kappa, theta, ak, at, X, Q = primals
        B, n = X.shape

        # 1) VJP through inner ActiveSetQPFunc (public API)
        # cotangents for outputs (x, active_comp, active_values, active_budget, budget_w)
        gQ_solve, gm1, gc, gwl, gwh, gL, gU, gp0, gx0 = (
            ActiveSetQPFunc.vjp_from_primals(
                tuple(inner_saved),
                v,
                None,
                None,
                None,
                None,
                needs_input_grad=None,
            )
        )

        # Route adjoints to original inputs through preprocessing:

        # Q: from inner plus from m1 = -m + p + 2 (X @ Q)  ⇒
        # ∂/∂Q_b = 2 * outer(X_b, gm1_b)
        gQ = gQ_solve + 2.0 * torch.einsum("bi,bj->bij", X, gm1)

        # m: m1 = -m + ...  ⇒ grad_m = -gm1
        gm = -gm1

        # c: passed through unchanged in our smooth path (no contribution)
        gc = torch.zeros_like(budget)

        # X:
        #   from m1: 2 * (Q^T @ gm1)
        #   from L,U: -gL, -gU
        #   from wl,wh: -(gwl + gwh) * p0
        gX = 2.0 * torch.bmm(Q.transpose(-1, -2), gm1.unsqueeze(-1)).squeeze(
            -1
        )
        gX = gX - gL - gU - (gwl + gwh).unsqueeze(1) * inner_saved[6]  # p0

        # budget:
        invp = (1.0 / p).unsqueeze(0)  # (1,n)
        gbudget = (
            (-(kappa.unsqueeze(1) * invp) * gL).sum(dim=1)
            + ((theta.unsqueeze(1) * invp) * gU).sum(dim=1)
            + (1.0 - theta) * gwl
            + (1.0 + kappa) * gwh
        )

        # kappa:
        gkappa = (-(budget.unsqueeze(1) * invp) * gL).sum(dim=1) + budget * gwh

        # theta:
        gtheta = ((budget.unsqueeze(1) * invp) * gU).sum(dim=1) - budget * gwl

        # p (market-wide vector):
        #   from m1: + sum_b gm1_b
        #   from p0: + sum_b gp0_b
        #   from L:  + sum_b gL_b * (budget*kappa)/p^2
        #   from U:  + sum_b gU_b * (-(budget*theta))/p^2
        #   from wl,wh: - sum_b (gwl_b + gwh_b) * X_b
        invp2 = (1.0 / (p * p)).unsqueeze(0)
        gp = (
            gm1.sum(dim=0)
            + gp0.sum(dim=0)
            + ((budget.unsqueeze(1) * kappa.unsqueeze(1) * invp2) * gL).sum(
                dim=0
            )
            - ((budget.unsqueeze(1) * theta.unsqueeze(1) * invp2) * gU).sum(
                dim=0
            )
            - ((gwl + gwh).unsqueeze(1) * X).sum(dim=0)
        )

        # x0: you treat the start projection as non-differentiable → zero
        gx0 = torch.zeros_like(x)

        grads = (gQ, gm, gc, gX, gbudget, gkappa, gtheta, gp, gx0)

        # Respect needs_input_grad
        if needs_input_grad is not None:
            grads = list(grads)
            for i, need in enumerate(needs_input_grad):
                if not need:
                    grads[i] = None
            grads = tuple(grads)
        return grads

    @staticmethod
    def jvp_from_primals(
        all_primals: Any, *all_tangents: tuple | None
    ) -> torch.Tensor:
        """
        JVP through:
          (Q, m, c, X, budget, kappa, theta, p, x0)
            -> (Q, m1, c, wl, wh, L, U, p0, x0)
            -> dx
        using ActiveSetQPFunc.jvp_from_primals.

        Return dx (B, n)
        """
        (dQ, dm, dc, dX, dbudget, dkappa, dtheta, dp, dx0) = all_tangents
        primals, inner_saved = StockSolverFunc._unpack_primals(all_primals)
        x, p, budget, kappa, theta, ak, at, X, Q = primals
        Q, m1, wl, wh, L, U, p0, *_ = inner_saved
        B, n = X.shape

        # Default missing tangents to zeros
        if dQ is None:
            dQ = torch.zeros_like(Q)
        if dm is None:
            dm = torch.zeros_like(m1)  # original m; will convert to dm1 below
        if dc is None:
            dc = torch.zeros_like(wl)
        if dX is None:
            dX = torch.zeros_like(X)
        if dbudget is None:
            dbudget = torch.zeros_like(budget)
        if dkappa is None:
            dkappa = torch.zeros_like(kappa)
        if dtheta is None:
            dtheta = torch.zeros_like(theta)
        if dp is None:
            dp = torch.zeros_like(p)
        if dx0 is None:
            dx0 = torch.zeros_like(x)

        # Broadcast helpers
        dp0 = dp.unsqueeze(0).expand(B, -1)  # (B, n)
        invp = (1.0 / p).unsqueeze(0)  # (1, n)
        invp2 = (1.0 / (p * p)).unsqueeze(0)  # (1, n)

        # m1 = -m + p + 2 * (X @ Q)
        dXQ = torch.bmm(dX.unsqueeze(1), Q).squeeze(1)
        XdQ = torch.bmm(X.unsqueeze(1), dQ).squeeze(1)
        dm1 = -dm + dp0 + 2.0 * (dXQ + XdQ)

        # L,U, wl, wh tangents from (budget, kappa, theta, p, X)
        dL = (
            -(
                dbudget.unsqueeze(1) * kappa.unsqueeze(1)
                + budget.unsqueeze(1) * dkappa.unsqueeze(1)
            )
            * invp
            + ((budget.unsqueeze(1) * kappa.unsqueeze(1)) * invp2) * dp0
            - dX
        )
        dU = (
            (
                dbudget.unsqueeze(1) * theta.unsqueeze(1)
                + budget.unsqueeze(1) * dtheta.unsqueeze(1)
            )
            * invp
            - ((budget.unsqueeze(1) * theta.unsqueeze(1)) * invp2) * dp0
            - dX
        )
        dsum = (X * dp0 + dX * p0).sum(dim=1)  # (B,)
        dwl = dbudget * (1.0 - theta) - budget * dtheta - dsum
        dwh = dbudget * (1.0 + kappa) + budget * dkappa - dsum

        # Inner JVP (public API)
        dx, *_ = ActiveSetQPFunc.jvp_from_primals(
            tuple(inner_saved),
            dQ,
            dm1,
            dc,
            dwl,
            dwh,
            dL,
            dU,
            dp0,
            torch.zeros_like(x),
        )
        return dx


def _to_float64_preserve_grad(x):
    """Cast to float64 without detaching; create tensor only if needed."""
    if isinstance(x, torch.Tensor):
        return x if x.dtype == torch.float64 else x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


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
    Optimize a single portfolio or a batch of portfolios under budget, short,
    and long leverage constraints.

    Batching
    --------
    Supports single or batched optimization. When batched, **all parameters
    except `prices` must be stacked along dimension 0** (the batch dimension).
    **`prices` is a market-wide vector shared by everyone and MUST
    be unbatched** (shape (N,)); it will be internally broadcast across the batch.

    The method accepts PyTorch tensors or any array-like objects convertible
    to `torch.Tensor`, and is fully differentiable (autograd-friendly).

    Parameters
    ----------
    Sigma : (N, N) or (B, N, N)
        Expected covariance matrix (PSD). Batched as (B, N, N).
    expected_returns : (N,) or (B, N)
        Expected returns. Batched as (B, N).
    commission : () or (B, )
        Per-agent commission to **buy**. Use zeros for initial portfolios.
    holdings : (N,) or (B, N)
        Current holdings (shares). Use zeros for initial portfolios.
    budget : () or (B,)
        Total budget (scalar or batched scalar).
    short_leverage : () or (N,) or (B,) or (B, N)
        Per-security short cap as a fraction of `budget`. `0` ⇒ no short sales.
    long_leverage : () or (N,) or (B,) or (B, N)
        Per-security long cap as a fraction of `budget`. Values < 1 ⇒ no leverage.
    prices : (N,)
        **Market prices** shared by all agents. **Must be 1-D (N,)**;
        batches are not allowed.

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
        expected_returns_t = _to_float64_preserve_grad(expected_returns).unsqueeze(0)
        return StockSolverFunc.apply(
            _to_float64_preserve_grad(Sigma).unsqueeze(0),
            _to_float64_preserve_grad(expected_returns).unsqueeze(0),
            _to_float64_preserve_grad(commission).unsqueeze(0),
            _to_float64_preserve_grad(holdings).unsqueeze(0),
            _to_float64_preserve_grad(budget).unsqueeze(0),
            _to_float64_preserve_grad(short_leverage).unsqueeze(0),
            _to_float64_preserve_grad(long_leverage).unsqueeze(0),
            _to_float64_preserve_grad(prices),
            torch.zeros_like(expected_returns_t).unsqueeze(0)
        )
    else:
        expected_returns_t = _to_float64_preserve_grad(expected_returns)
        return StockSolverFunc.apply(
            _to_float64_preserve_grad(Sigma),
            _to_float64_preserve_grad(expected_returns),
            _to_float64_preserve_grad(commission),
            _to_float64_preserve_grad(holdings),
            _to_float64_preserve_grad(budget),
            _to_float64_preserve_grad(short_leverage),
            _to_float64_preserve_grad(long_leverage),
            _to_float64_preserve_grad(prices),
            torch.zeros_like(expected_returns_t)
        )


class StockSolverSum(ExplicitADFunction):
    """
    Sum-reduction wrapper around StockSolverFunc:
      y = sum_b StockSolverFunc(Q, m, c, X, budget, kappa, theta, p, x0)[b, :]
    Reuses StockSolverFunc.{compute, compute_primals, vjp_from_primals, jvp_from_primals}.
    """

    # ---------- Interface required by ExplicitADFunction ----------

    @staticmethod
    def compute(
        Q: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        X: torch.Tensor,
        budget: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        p: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward: call StockSolverFunc.compute to get x (B, n), then sum over dim=0 -> (n,)
        """
        x = StockSolverFunc.compute(
            Q, m, c, X, budget, kappa, theta, p, x0
        )  # (B, n)
        return x.sum(dim=0)  # (n,)

    @staticmethod
    def compute_primals(
        *inputs: torch.Tensor, outputs: Tuple[torch.Tensor, ...] | torch.Tensor
    ) -> Any:
        """
        Primals: just delegate to StockSolverFunc.compute_primals so we can reuse its
        saved structure for VJP/JVP. We ignore our own reduced `outputs`.
        """
        # NOTE: StockSolverFunc.compute_primals returns a tuple that starts with
        # (x, p, budget, kappa, theta, ak, at, X, ...) and includes everything
        # needed to run inner vjp/jvp without autograd tracing.
        return StockSolverFunc.compute_primals(*inputs, outputs=outputs)

    @staticmethod
    def vjp_from_primals(
        saved: Any,
        *cotangents: torch.Tensor,
        needs_input_grad: Sequence[bool] | None = None,
    ) -> Tuple[torch.Tensor | None, ...]:
        """
        Backward (VJP): Our output is y = sum_b x_b, so the cotangent
        for each x_b is the same v. Delegate inner VJP to StockSolverFunc and
        return grads for (Q, m, c, X, budget, kappa, theta, p, x0).
        """
        # Unpack batch size B and feature size n from saved
        # (X is at index 7 in StockSolverFunc saved)
        # saved layout begins with: (x, p, budget, kappa, theta, ak, at, X, ...)
        X = saved[7]  # (B, n)
        B, _ = X.shape

        (v,) = cotangents  # v is shape (n,)
        v_batched = v.unsqueeze(0).expand(B, -1).contiguous()  # (B, n)

        # Inner VJP expects cotangents for its 5 outputs:
        # (x, active_comp, active_values, active_budget, budget_w)
        inner_grads = StockSolverFunc.vjp_from_primals(
            saved, v_batched, None, None, None, None, needs_input_grad=None
        )
        # inner_grads order: (gQ, gm, gc, gX, gbudget, gkappa, gtheta, gp, gx0)
        grads = inner_grads

        # Respect needs_input_grad mask if provided
        # (ExplicitADFunction will also mask, but do it here too)
        if needs_input_grad is not None:
            grads = list(grads)
            for i, need in enumerate(needs_input_grad):
                if not need:
                    grads[i] = None
            grads = tuple(grads)
        return grads

    @staticmethod
    def jvp_from_primals(
        saved: Any, *tangents: torch.Tensor | None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward-mode (JVP): inner JVP gives dx (B, n).
        Our output tangent is sum over batch -> (n,).
        """
        # Inner JVP (same tangent ordering as inputs):
        # (dQ, dm, dc, dX, dbudget, dkappa, dtheta, dp, dx0)
        dx_batched = StockSolverFunc.jvp_from_primals(
            saved, *tangents
        )  # (B, n)
        dy = dx_batched.sum(dim=0)  # (n,)
        return (dy,)

    """
    GMRES Solver
    """

    @staticmethod
    def matvec(x_flat: torch.Tensor, mode, primals):
        """
        GMRES wrapper that can apply either J (via jvp) or J^T (via vjp) for StockSolverSum.
        """
        # primals here is exactly whatever ImplicitFunction gave to gmres:
        # typically the `primals_F` from func.compute_primals(...)
        saved = primals[0] if isinstance(primals, tuple) and len(primals) == 1 else primals

        # StockSolverSum's saved primals start with StockSolverPrimals:
        # (x, p, budget, kappa, theta, ak, at, X, Q, ...)
        p = saved[1]         # (n,) for shared price vector
        n = p.shape[-1]
        v = x_flat.view(n)   # Jacobian acts on shape (n,)

        if mode == 0:
            # J @ v  via forward-mode
            tangents = (None, None, None, None, None, None, None, v, None)
            (dy,) = StockSolverSum.jvp_from_primals(saved, *tangents)
            return dy.reshape(-1)
        elif mode == 1:
            # J^T @ v via reverse-mode
            grads = StockSolverSum.vjp_from_primals(saved, v, needs_input_grad=None)
            g_p = grads[7]   # gradient w.r.t. price input
            return g_p.reshape(-1)
        else:
            raise ValueError(f"Unknown GMRES kind")


    @staticmethod
    def compute_residual(b, x, mode, primals):
        return b - StockSolverSum.matvec(x, mode, primals)

    @staticmethod
    def _safe_normalize(x: torch.Tensor):
        """
        L2-normalize x; if ||x|| <= thresh, return (0 vector, 0).
        """
        thresh = torch.finfo(x.dtype).eps
        thresh_t = torch.as_tensor(thresh, dtype=x.dtype, device=x.device)
        return StockSolverSum._safe_normalize_thresh(x, thresh_t)

    @staticmethod
    def _safe_normalize_thresh(x: torch.Tensor, thresh: torch.Tensor):
        norm = torch.linalg.vector_norm(x)  # scalar tensor

        # Avoid div-by-zero by clamping the denominator
        safe_norm = torch.clamp(norm, min=thresh)

        y = x / safe_norm  # well-defined
        use = norm > thresh

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

    @staticmethod
    def _kth_arnoldi_iteration(
        k: torch.Tensor,  # scalar int tensor
        V: torch.Tensor,  # (m, restart+1)
        H: torch.Tensor,  # (restart, restart+1)
        mode,
        primals,  # tuple
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
        w = StockSolverSum.matvec(v_k, mode, primals)
        _, w_norm0 = StockSolverSum._safe_normalize(w)

        # Project against ALL columns in V (unused ones are zeros, so harmless)
        q, r = StockSolverSum._iterative_classical_gram_schmidt(
            V, w
        )  # r: (ncols,)

        tol = (
            torch.as_tensor(torch.finfo(dtype).eps, dtype=dtype, device=device)
            * w_norm0
        )
        unit_v, v_norm_1 = StockSolverSum._safe_normalize_thresh(q, tol)

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
    def _loop_cond(V, H, breakdown, k, restart_shape, mode, *_):
        restart = restart_shape.shape[0]
        return (k < restart) & (~breakdown)

    @staticmethod
    def _arnoldi_process(V, H, breakdown, k, restart_shape, mode, *primals):
        V2, H2, breakdown2 = StockSolverSum._kth_arnoldi_iteration(k, V, H, mode, primals)
        return V2, H2, breakdown2, k + 1

    @staticmethod
    def _gmres_batched(
        b, x0, unit_residual, residual_norm, restart_shape, mode, primals
    ):
        device, dtype = b.device, b.dtype
        m = b.shape[0]
        restart = restart_shape.shape[0]
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

        V, H, _, k = ops.higher_order.while_loop(
            arnoldi_cond_fun,
            arnoldi_body_fun,
            (V0, H0, br0, k0),
            (restart_shape, mode) + primals,
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
        residual = StockSolverSum.compute_residual(b, x, mode, primals)

        unit_residual_new, residual_norm_new = StockSolverSum._safe_normalize(
            residual
        )
        return x, unit_residual_new, residual_norm_new

    @staticmethod
    def _gmres_cond_fun(x, k, ures, rnorm, b, maxiter, thresh, *_):
        return (k < maxiter) & (rnorm > thresh)

    @staticmethod
    def _gmres_body_fun(
        x, k, ures, rnorm, b, maxiter, thresh, restart_shape, mode, *primals
    ):
        x_new, ures_new, rnorm_new = StockSolverSum._gmres_batched(
            b, x, ures, rnorm, restart_shape, mode, primals
        )
        return x_new, k + 1, ures_new, rnorm_new

    @staticmethod
    def gmres_x0(
        b: torch.Tensor,
        x0: torch.Tensor,
        primals,
        tol: float = 1e-5,
        atol: float = 0.0,
        restart: int = 20,
        mode: int = 0, # 0 or 1
        maxiter: int = 0,
    ):
        assert b.ndim == 1, "This implementation expects 1D vectors."
        assert mode == 0 or mode ==1, "Unknown GMRES mode"        
        device, dtype = b.device, b.dtype
        m = b.shape[0]

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
        residual0 = StockSolverSum.compute_residual(b, x0, 0, primals)
        unit_residual, residual_norm = StockSolverSum._safe_normalize(residual0)

        # while_loop state must be tensors
        k0 = torch.tensor(0, dtype=torch.int64, device=device)

        maxiter_t = torch.as_tensor(maxiter, device=device, dtype=torch.int64)
        default_maxiter = torch.as_tensor(10 * m, device=device, dtype=torch.int64)
        maxiter_t = torch.where(maxiter_t == 0, default_maxiter, maxiter_t)

        mode_t = torch.tensor(mode, dtype=torch.int64, device=device)
        restart_shape = torch.zeros((restart,), device=device)

        x_final, *_ = ops.higher_order.while_loop(
            gmres_cond_fun,
            gmres_body_fun,
            (x0, k0, unit_residual, residual_norm),
            (b, maxiter_t, thresh, restart_shape, mode_t) + primals,
        )
        return x_final

    @staticmethod
    def gmres(
        b: torch.Tensor,
        primals,
        tol: float = 1e-5,
        atol: float = 0.0,
        restart: int = 20,
        mode: int = 0, # 0 or 1
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
        b : torch.Tensor
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
        mode : integer, optional
            0 for JVP, 1 for VJP
        maxiter : integer
            Maximum number of times to rebuild the size-``restart`` Krylov space
            starting from the solution found at the last iteration. If GMRES
            halts or is very slow, decreasing this parameter may help.
            Default is infinite.
        """

        x0 = torch.zeros_like(b)
        return StockSolverSum.gmres_x0(b, x0, primals, tol, atol, restart, mode, maxiter)


# ---------- GMRES outer loop (Newton/GMRES) ----------

def gmres_cond_fun(x, k, ures, rnorm, b, maxiter, thresh, *rest):
    """
    Wrapper for StockSolverSum._gmres_cond_fun so TorchDynamo sees
    a plain function instead of a class attribute.
    """
    return StockSolverSum._gmres_cond_fun(
        x, k, ures, rnorm, b, maxiter, thresh, *rest
    )


def gmres_body_fun(x, k, ures, rnorm, b, maxiter, thresh, restart, mode, *primals):
    """
    Wrapper for StockSolverSum._gmres_body_fun.
    """
    return StockSolverSum._gmres_body_fun(
        x, k, ures, rnorm, b, maxiter, thresh, restart, mode, *primals
    )


# ---------- Inner Arnoldi loop for GMRES Krylov basis ----------

def arnoldi_cond_fun(V, H, breakdown, k, restart, *rest):
    """
    Wrapper for StockSolverSum._loop_cond used inside the Arnoldi while_loop.
    """
    return StockSolverSum._loop_cond(
        V, H, breakdown, k, restart, *rest
    )


def arnoldi_body_fun(V, H, breakdown, k, restart, mode, *primals):
    """
    Wrapper for StockSolverSum._arnoldi_process used inside the Arnoldi while_loop.
    """
    return StockSolverSum._arnoldi_process(
        V, H, breakdown, k, restart, mode, *primals
    )


class ImplicitFunction(ExplicitADFunction):
    """
    Solve for one *variable input* x_var in a vector equation

        F(*inputs) = y_target

    using a Newton–Krylov method with GMRES-based linear solves, and expose
    reverse-mode (VJP) for the implicit map g: (y_target, *inputs_fixed) -> x_var.

    Usage pattern (important):
      - Subclass this class and set:
          func = <ExplicitADFunction subclass that also provides GMRES hooks>
          variable_input = <1-based index of the input to solve for>
      - Call .apply with *varargs* (no tuples!) so autograd can see all inputs:
          x_var = MyImplicit.apply(y_target, in1, in2, ..., inN)

    Requirements on `func`:
      - Implements ExplicitADFunction-style API:
          compute(*inputs) -> Tensor
          compute_primals(*inputs, outputs=Tensor) -> saved
          jvp_from_primals(saved, *tangents_per_input) -> Tensor
          vjp_from_primals(saved, bar_y, needs_input_grad=None) -> tuple(grads_per_input)
        (Single-tensor output is assumed. If your F returns a tuple of outputs,
         adapt this class: a single-output is required for GMRES right-hand side.)
      - Provides GMRES interface via inheritance from your StockSolverSum:
          matvec(x, primals) -> A @ x
          compute_residual(b, x, primals) -> b - A @ x
          gmres(b, primals, x0=None, tol=..., atol=..., restart=..., maxiter=...) -> x

    Contract with autograd:
      - .apply is called as  x_var = ImplicitSubclass.apply(y_target, *inputs)
      - vjp_from_primals must return (grad_wrt_y, *grads_wrt_inputs) to match the
        number of *tensor* args passed to .apply.
    """

    # You MUST set these in a subclass, e.g.:
    #   func = MyFunctionGMRES
    #   variable_input = 3  # 1-based index of the variable among *inputs
    func = None
    variable_input: int | None = None  # 1-based index

    # Newton–Krylov parameters
    _newton_maxiter = 50
    _newton_tol = 1e-10

    # GMRES parameters (used for inner linear solves)
    _gmres_restart = 20
    _gmres_tol = 1e-8
    _gmres_atol = 0.0
    _gmres_maxiter = 200

    # Backtracking line-search parameters
    _ls_min_step = 1e-6  # minimal step size alpha before we terminate
    _ls_max_halves = (
        None  # None ⇒ unlimited until _ls_min_step, or set an int cap
    )

    # ------------------------------ helpers ------------------------------

    @staticmethod
    def _unwrap_single(x):
        """Accept either a Tensor or a (Tensor,) singleton; raise otherwise."""
        if isinstance(x, tuple):
            if len(x) != 1:
                raise TypeError(
                    "ImplicitFunction expects `func` to be single-output (Tensor). "
                    f"Got a tuple of length {len(x)}."
                )
            return x[0]
        return x

    # -------------------- ExplicitADFunction: forward ---------------------

    @classmethod
    def compute(cls, y_target: torch.Tensor, *inputs: torch.Tensor):
        """
        Solve F(*inputs) = y_target for inputs[var_i], where var_i = variable_input - 1.

        Returns the solved variable tensor (same shape as y_target).
        """
        if cls.func is None:
            raise RuntimeError(
                "ImplicitFunction.func must be set to an ExplicitADFunction+GMRES class."
            )
        if not isinstance(cls.variable_input, int):
            raise TypeError(
                "ImplicitFunction.variable_input must be a 1-based integer index."
            )
        if len(inputs) == 0:
            raise ValueError("At least one input is required.")

        var_i = cls.variable_input - 1
        if not (0 <= var_i < len(inputs)):
            raise IndexError(
                f"variable_input={cls.variable_input} is out of range "
                f"for {len(inputs)} inputs."
            )

        xs = list(inputs)
        x_var = xs[var_i]
        if not isinstance(x_var, torch.Tensor):
            raise TypeError("Variable input must be a Tensor.")
        # Square local system: shape(x_var) == shape(y)
        if x_var.shape != y_target.shape:
            raise ValueError(
                "Assuming square local system: "
                "shape(variable input) must equal shape(y_target)."
            )

        device, dtype = y_target.device, y_target.dtype
        Func = cls.func  # capture subclass' func for nested classes

        for _ in range(cls._newton_maxiter):
            # Residual at current iterate
            y_cur = cls._unwrap_single(Func.compute(*xs))
            r = y_cur - y_target
            r_before = torch.linalg.vector_norm(r)
            if r_before <= torch.as_tensor(
                cls._newton_tol, device=device, dtype=dtype
            ):
                break

            # Linearize at current point and build J_var via JVP
            primals = Func.compute_primals(*xs, outputs=y_cur)
            x_shape = x_var.shape

            # Solve J Δ = -r
            dx_flat = StockSolverSum.gmres(
                b=(-r).reshape(-1),
                primals=primals,
                tol=cls._gmres_tol,
                atol=cls._gmres_atol,
                restart=cls._gmres_restart,
                mode=0,
                maxiter=cls._gmres_maxiter,
            )
            dx = dx_flat.view_as(x_var)

            # Backtracking line search (halve alpha until residual decreases)
            alpha = torch.tensor(1.0, device=device, dtype=dtype)
            halved = 0
            while True:
                x_trial = x_var + alpha * dx
                xs_trial = list(xs)
                xs_trial[var_i] = x_trial
                y_trial = cls._unwrap_single(Func.compute(*xs_trial))
                r_after = torch.linalg.vector_norm(y_trial - y_target)

                if r_after <= r_before:  # accept
                    x_var, xs = x_trial, xs_trial
                    break

                alpha = alpha * 0.5
                halved += 1
                if alpha.item() < cls._ls_min_step or (
                    cls._ls_max_halves is not None
                    and halved >= cls._ls_max_halves
                ):
                    # terminate: step too small and still not decreasing residual
                    return x_var

        return x_var

    # ----------------- ExplicitADFunction: cache primals ------------------

    @classmethod
    def compute_primals(
        cls,
        y_target: torch.Tensor,
        *inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        """
        Prepare and cache primals at the solution point for implicit VJP.

        Arguments match the autograd plumbing:
          - y_target, *inputs  : the same args passed to .compute(...)
          - outputs            : the x_var returned by .compute(...)
        """
        var_i = cls.variable_input - 1
        xs_sol = list(inputs)
        xs_sol[var_i] = outputs.detach()

        y_sol = cls._unwrap_single(cls.func.compute(*xs_sol))
        primals_F = cls.func.compute_primals(*xs_sol, outputs=y_sol)

        return {
            "y": y_target.detach(),
            "inputs_sol": tuple(t.detach() for t in xs_sol),
            "primals_F": primals_F,
            "var_i": var_i,
            "shape_y": y_sol.shape,
            "shape_xvar": outputs.shape,
        }

    # ----------------- ExplicitADFunction: implicit VJP -------------------

    @classmethod
    def vjp_from_primals(
        cls, saved, bar_xvar: torch.Tensor, needs_input_grad=None
    ):
        """
        Given upstream cotangent on the output (x_var), return gradients wrt
        (y_target, *inputs) for the implicit map.

        Implicit VJP:
          1) Solve (∂F/∂x_var)^T w = bar_xvar
          2) bar_y = w
          3) bar_inputs[j] = - (∂F/∂x_j)^T w  for j != var_i
             bar_inputs[var_i] = None   (the initial guess is not a true input to g)
        """
        primals_F = saved["primals_F"]
        var_i = saved["var_i"]
        shape_y = saved["shape_y"]
        Func = cls.func

        # Solve J^T w = bar_xvar
        w_flat = StockSolverSum.gmres(
            b=bar_xvar.reshape(-1),
            primals=primals_F,
            tol=cls._gmres_tol,
            atol=cls._gmres_atol,
            restart=cls._gmres_restart,
            mode=1,
            maxiter=cls._gmres_maxiter,
        )
        w = w_flat.view(shape_y)

        # Gradients w.r.t. inputs via VJP through func at the solution point
        full_vjp = Func.vjp_from_primals(
            primals_F, w
        )  # tuple of length == len(inputs)
        bar_inputs = [
            (None if j == var_i else (-g if g is not None else None))
            for j, g in enumerate(full_vjp)
        ]
        bar_y = w

        if needs_input_grad is not None:
            # needs_input_grad corresponds to (y_target, *inputs)
            grads_all = (bar_y, *bar_inputs)
            masked = tuple(
                g if need else None
                for g, need in zip(grads_all, needs_input_grad)
            )
            return masked

        return (bar_y, *bar_inputs)


class PriceSolver(ImplicitFunction):
    func = StockSolverSum
    variable_input = 8


def _to_float64_preserve_grad(x):
    """Cast to float64 without detaching; create tensor only if needed."""
    if isinstance(x, torch.Tensor):
        return x if x.dtype == torch.float64 else x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


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
    Sigma = _to_float64_preserve_grad(Sigma)
    mu = _to_float64_preserve_grad(expected_returns)
    com = _to_float64_preserve_grad(commission)
    X = _to_float64_preserve_grad(holdings)
    budget = _to_float64_preserve_grad(budget)
    kappa = _to_float64_preserve_grad(short_leverage)
    theta = _to_float64_preserve_grad(long_leverage)
    S = _to_float64_preserve_grad(supply)
    p0 = (
        torch.ones_like(S)
        if initial_approximation is None
        else _to_float64_preserve_grad(initial_approximation)
    )
    x0 = torch.zeros_like(mu)
    # y_target is supply; inputs contain the *initial* prices in the variable slot
    return PriceSolver.apply(
        S, Sigma, mu, com, X, budget, kappa, theta, p0, x0
    )
