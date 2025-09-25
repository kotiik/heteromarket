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

### Example 1

```python
from heteromarket import optimize_portfolio
import torch
import numpy as np

# expected prices
M = np.array([50.15707481, 49.95627699, 50.2048171])
# real prices
prices = np.array([50.0, 50.0, 50.0])
# covariance matrix
Sigma = np.array(
    [[0.0737, 0.0354, 0.0183], 
     [0.0354, 0.0572, 0.0242], 
     [0.0183, 0.0242, 0.0153]]
)

# Single portfolio
h = optimize_portfolio(
    Sigma, M,
    commission=0.0,
    holdings=np.zeros(3),
    budget=1019.2129,
    short_leverage=0.1363,
    long_leverage=0.6878,
    prices=prices
)

print("Optimal portfolio : ", h)
```

Expected result:

```
Optimal portfolio :  tensor([[-0.3851, -2.7784,  9.5275]], dtype=torch.float64)
```

---

### Example 2

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

### Example 1

```python
from heteromarket import find_equilibrium_prices
import torch
import pyro
import pyro.distributions as dist

n, B = 3, 20
pyro.set_rng_seed(42)

V0 = torch.eye(n) * 50.0
m0 = torch.ones(n)
df = 4
k0 = 10.0
# c_a = 2.0
# c_scale = 0.005
budget_b = 10.0
budget_scale = 130000.0
kappa_a = 1.0
kappa_b = 4.0
theta_a = 2.0
theta_b = 2.0
lam_a = 1.0
lam_scale = 0.1

scale = V0 / df
scale_tril = torch.linalg.cholesky(scale)

# --- Priors ---
with pyro.plate("B", B):
    M = pyro.sample("M", dist.MultivariateNormal(m0, covariance_matrix=V0 / k0))
    Sigma = pyro.sample(
        "Sigma",
        dist.Wishart(df=df, scale_tril=scale_tril),
    )
    c = torch.zeros((B,))
    budget = pyro.sample("budget", dist.Pareto(budget_scale, budget_b))
    kappa = pyro.sample("kappa", dist.Beta(kappa_a, kappa_b))
    theta = pyro.sample("theta", dist.Beta(theta_a, theta_b)) / n * (n - 1) + 1 / n
    S = torch.ones(n) * 10000.0

x_opt = find_equilibrium_prices(Sigma, M, c, torch.zeros_like(M), budget, kappa, theta, S)
print("Equilibrium prices:", x_opt)
```

expected output:
```
Equilibrium prices: tensor([29.2456, 32.4128, 20.9177], dtype=torch.float64)
```
---

### Example 2

```python
from heteromarket import find_equilibrium_prices
import torch
import pyro
import pyro.distributions as dist

n, B = 3, 20
pyro.set_rng_seed(42)

V0 = torch.eye(n) * 50.0
m0 = torch.ones(n)
df = 4
k0 = 10.0

budget_b = 10.0
budget_scale = 130000.0
kappa_a = 1.0
kappa_b = 4.0
theta_a = 2.0
theta_b = 2.0

scale = V0 / df
scale_tril = torch.linalg.cholesky(scale)

# --- Priors ---
with pyro.plate("B", B):
    M = pyro.sample("M", dist.MultivariateNormal(m0, covariance_matrix=V0 / k0))
    Sigma = pyro.sample(
        "Sigma",
        dist.Wishart(df=df, scale_tril=scale_tril),
    )
    c = torch.zeros((B,))
    budget = pyro.sample("budget", dist.Pareto(budget_scale, budget_b))
    kappa = pyro.sample("kappa", dist.Beta(kappa_a, kappa_b))
    theta = pyro.sample("theta", dist.Beta(theta_a, theta_b)) / n * (n - 1) + 1 / n
    S = torch.ones(n) * 10000.0

Sigma.requires_grad_(True)
M.requires_grad_(True)
x_opt = find_equilibrium_prices(Sigma, M, c, torch.zeros_like(M), budget, kappa, theta, S)
x_opt[0].backward()

print('Gradient of Sigma[0] :\n',Sigma.grad[0])
print('Gradient of M :\n',M.grad)
```


---

## Key Points

* **Market prices (`prices`) are unbatched** in `optimize_portfolio` (shared across all agents).
* **All inputs must be batched** in `find_equilibrium_prices`.
* Both functions are **autograd-friendly**.
* Designed for use in **optimization pipelines, calibration, and Bayesian inference**.

