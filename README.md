# heteromarket

**heteromarket** is a Python package for solving heterogeneous agent market models.
It provides two main functions:

1. **`optimize_portfolio`** – optimize individual (or batched) portfolios under budget and leverage constraints.
2. **`find_equilibrium_prices`** – compute market-clearing prices for heterogeneous agents.

Both functions accept **PyTorch tensors** (or array-likes convertible to tensors) and are **fully differentiable**, so they can be embedded in optimization, calibration, or inference pipelines (e.g. MCMC NUTS).

## installation

```
!pip install "git+https://github.com/kotiik/heteromarket.git"
```

---

## `optimize_portfolio`

Optimize a single portfolio or a batch of portfolios under budget, short, and long leverage constraints.

* Supports **single or batched** inputs.
* **All inputs except `prices` can be batched** along dimension 0.
* `prices` is **market-wide** and must be unbatched (shape `(N,)`).

### Parameters

* **`Sigma`** `(N, N)` or `(B, N, N)`: Expected covariance matrix of asset returns.
* **`expected_returns`** `(N,)` or `(B, N)`: Expected returns.
* **`commission`** `(N,)` or `(B, N)`: Per-asset buy commission. Use zeros for initial portfolio.
* **`holdings`** `(N,)` or `(B, N)`: Current holdings. Use zeros for initial portfolio.
* **`budget`** `()` or `(B,)`: Total budget (scalar or batched).
* **`short_leverage`** scalar, vector, or batched: Short-sale cap (fraction of budget). `0` = no shorts.
* **`long_leverage`** scalar, vector, or batched: Long position cap (fraction of budget). `< 1` = no leverage.
* **`prices`** `(N,)`: Market prices (shared, unbatched).

### Returns

* **`optimal_holdings`** `(N,)` or `(B, N)`: Optimal post-trade holdings.

### Example

```python
from heteromarket import optimize_portfolio
import torch

# Single portfolio
h = optimize_portfolio(
    Sigma, mu,
    commission=torch.zeros_like(mu),
    holdings=torch.zeros_like(mu),
    budget=1.0,
    short_leverage=0.0,
    long_leverage=1.0,
    prices=prices
)

# Batched portfolios (B, N) with shared market prices (N,)
h_b = optimize_portfolio(
    Sigma_b, mu_b, commission_b, holdings_b,
    budget_b, short_leverage=0.0, long_leverage=1.0,
    prices=prices
)
```

---

## `find_equilibrium_prices`

Compute market-clearing (equilibrium) prices for a batch of heterogeneous-agent markets.

* **Batched only**: all inputs except prices must have leading dimension `B`.
* `initial_approximation` is optional, must be positive, and only affects convergence speed (not results).

### Parameters

* **`Sigma`** `(B, N, N)`: Expected covariance matrices.
* **`expected_returns`** `(B, N)`: Expected returns.
* **`commission`** `(B, N)`: Per-asset buy commission. Use zeros for initial portfolio.
* **`holdings`** `(B, N)`: Current holdings. Use zeros for initial portfolio.
* **`budget`** `(B,)`: Budget per batch item.
* **`short_leverage`** `(B,)` or `(B, N)`: Short-sale cap (fraction of budget).
* **`long_leverage`** `(B,)` or `(B, N)`: Long position cap (fraction of budget).
* **`supply`** `(N,)`: Exogenous asset supply.
* **`initial_approximation`** `(N,)`, optional: Positive initial guess for prices. Improves performance if close to equilibrium.

### Returns

* **`equilibrium_prices`** `(B, N)`: Market-clearing prices.

### Example

```python
from heteromarket import find_equilibrium_prices
import torch

p_eq = find_equilibrium_prices(
    Sigma_b, mu_b,
    commission_b, holdings_b, budget_b,
    short_leverage_b, long_leverage_b,
    supply_b,
    initial_approximation=torch.ones_like(supply_b)
)
```

---

## Key Points

* **Market prices (`prices`) are unbatched** in `optimize_portfolio` (shared across all agents).
* **All inputs must be batched** in `find_equilibrium_prices`.
* Both functions are **autograd-friendly**.
* Designed for use in **optimization pipelines, calibration, and Bayesian inference**.

