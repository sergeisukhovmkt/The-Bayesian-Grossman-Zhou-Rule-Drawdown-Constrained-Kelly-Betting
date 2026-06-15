# Bayesian Grossman-Zhou: Drawdown-Constrained Kelly Betting with Parameter Uncertainty

**Working Paper · Sukhov (2026c) · Market Microstructure Research Lab**

---

## Overview

This repository accompanies the paper:

> **The Bayesian Grossman-Zhou Rule: Drawdown-Constrained Kelly Betting with Parameter Uncertainty**  
> Sergei Sukhov · Market Microstructure Research Lab · June 2026
> [`SSRN`](https://papers.ssrn.com/abstract=6942459)

### Problem

The Kelly criterion prescribes leverage $f = \kappa = (\mu - r)/\sigma^2$ as if the drift $\mu$ were known. In practice, $\mu$ must be estimated from data, and trading mandates impose hard drawdown barriers (typically 15–20% from peak). These two frictions interact: overestimation of $\mu$ causes overleverage precisely when the barrier is most binding.

### Main Result

Under the undiscounted growth criterion, the joint HJB in state space $(d_t, n_t)$ — where $d_t$ is the log-distance to the drawdown barrier and $n_t$ is the controlled effective sample size from the Kalman-Bucy filter — yields a tractable stationary benchmark:

$$f^{\text{BGZ}}(d, n) = \bar\kappa(n) \cdot \frac{d}{b}$$

where $\bar\kappa(n) = (\bar\mu - r)/\sigma^2$ is the Bayesian shrinkage estimate of the Kelly fraction and $b = -\ln(1 - \delta)$.

This **Bayesian Grossman-Zhou (Bayes GZ)** rule nests the classical Grossman-Zhou linear cushion rule at $n \to \infty$ and reduces leverage toward zero as either $d \to 0$ (barrier approach) or $n \to 0$ (data scarcity).

---

## Key Results

### Theoretical

| Result | Statement |
|--------|-----------|
| **Proposition 1** (Bayes GZ) | Bayes GZ is the tractable stationary benchmark under $r = 0$ |
| **Theorem 1** (Non-separability) | No separable $f = g(n) \cdot d$ solves the HJB for $r > 0$ |
| **Theorem 2** (Asymptotic) | Bayes GZ is the leading-order term in the $r/\mu$ expansion |
| **Proposition** (Double prudence) | $\partial^2 f / \partial d \, \partial n = \bar\kappa'(n)/b > 0$ — barrier distance and data quality are complements |
| **Proposition** (Convergence) | $D(n) = \kappa - \bar\kappa(n) = (\kappa - \kappa_0) \cdot n_0/(n_0 + n)$ — exact $O(1/n)$ |

### Empirical (N = 50,000 GBM paths)

| Scenario | Plug-in $U(\pi)$ | Bayes GZ $U(\pi)$ | $\Delta$Surv |
|----------|-----------------|-------------------|--------------|
| Oracle ($n_{\text{obs}} \to \infty$) | +0.083 | +0.072 | 0 pp |
| 5 yr data, $\sigma_u = 0.02$ | −0.003 | −0.010 | −0.2 pp |
| 2 yr data, $\sigma_u = 0.03$ | −0.169 | −0.084 | +4.2 pp |
| **1 yr data, $\sigma_u = 0.04$** | **−0.337** | **−0.104** | **+11.2 pp** |

### Stress Tests (5 return environments)

| Environment | $\Delta U$ (BGZ − plug-in) | Winner |
|-------------|--------------------------|--------|
| A. GBM baseline | +0.085 | Bayes GZ |
| B. Markov regime switching | +0.057 | Bayes GZ |
| C. Student-$t$ fat tails ($\nu = 4$) | +0.039 | Bayes GZ |
| D. Jump diffusion (Merton) | −0.041 | Plug-in |
| E. GARCH volatility clustering | −0.082 | Plug-in |

Bayes GZ dominates in environments A–C. Under jump diffusion and GARCH clustering (D–E), the conservative leverage reduces growth without improving survival; an extended model with jump-aware barrier or volatility-state variable would be needed.

---

## Repository Structure

```
.
├── code/
│   ├── simulation.py          # Main simulation engine (5 strategies, vectorised)
│   ├── stress_tests.py        # 5 return-environment stress tests
│   ├── generate_figures.py    # Figures 1–6 (policy shapes, MC, robustness)
│   └── fig7_stress_tests.py   # Figure 7 (stress test comparison)
│
├── figures/
│   ├── fig1_policy_shapes.pdf
│   ├── fig2_value_function.pdf
│   ├── fig3_double_prudence.pdf
│   ├── fig4_error_bounds.pdf
│   ├── fig5_monte_carlo.pdf
│   ├── fig6_robustness.pdf
│   └── fig7_stress_tests.pdf
│
├── data/                      # Simulation outputs (generated, not tracked)
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick check (N = 5,000 paths, ~10 seconds)
make fast

# 3. Full simulation (N = 50,000 paths, ~3 minutes)
make simulate

# 4. Stress tests across 5 return environments
make stress

# 5. Regenerate all figures
make figures
```

---

## Usage

### Compute Bayes GZ leverage

```python
import numpy as np

# Parameters
MU0, N0 = 0.05, 10.0   # prior: mean drift, effective sample size
SIG      = 0.20          # volatility
DELTA    = 0.20          # drawdown threshold
B        = -np.log(1 - DELTA)   # log-barrier width ≈ 0.2231

def kbar(n_obs_years: float, mu_hat: float) -> float:
    """Bayesian shrinkage estimate of Kelly fraction."""
    n = 12.0 * n_obs_years
    mu_post = (N0 * MU0 + n * mu_hat) / (N0 + n)
    return mu_post / SIG**2

def f_bgz(d: float, n_obs_years: float, mu_hat: float) -> float:
    """
    Bayes GZ leverage: f^BGZ(d, n) = kbar(n) * d / b
    
    Parameters
    ----------
    d            : current log-distance to drawdown barrier, in (0, b]
    n_obs_years  : years of pre-trade estimation data
    mu_hat       : estimated drift (from historical data)
    """
    return kbar(n_obs_years, mu_hat) * d / B

# Example: 2 years of data, mu_hat = 7.5%, current drawdown buffer = 15%
d_current   = -np.log(1 - 0.15)   # ≈ 0.163  (15% buffer from peak)
n_obs       = 2.0
mu_estimate = 0.075

leverage = f_bgz(d_current, n_obs, mu_estimate)
print(f"Bayes GZ leverage: {leverage:.3f}")   # ≈ 0.58
```

### Run the full simulation

```python
from code.simulation import BayesGZ, simulate

result = simulate(
    rule        = BayesGZ(),
    N_paths     = 10_000,
    n_obs_years = 2.0,
    sigma_u     = 0.03,
    seed        = 42,
)
print(f"U(π) = {result['penalised_utility']:+.4f}")
print(f"Survival = {result['survival_rate']*100:.1f}%")
```

---

## Model Parameters (Baseline)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Drift | $\mu$ | 0.08 | True annual drift |
| Volatility | $\sigma$ | 0.20 | Annual volatility |
| Drawdown threshold | $\delta$ | 0.20 | 20% from peak |
| Risk-free rate | $r$ | 0.00 | (simulation) |
| HJB discount rate | $r$ | 0.01 | (asymptotic theory) |
| Horizon | $T$ | 2 yr | Trading horizon |
| Step | $\Delta t$ | 1/252 | Daily |
| Prior mean | $\mu_0$ | 0.05 | Conservative prior |
| Prior strength | $n_0$ | 10 | Equivalent months |
| Ruin penalty | $p$ | −2 | Penalised utility |

---

## Revision History

| Version | Date | Change |
|---------|------|--------|
| v1.0 | Apr 2026 | Initial draft; proposed $f^*(d,n) = \bar\kappa(n)\,d/(1-\alpha(n))$ for $r > 0$ |
| v1.1 | Apr 2026 | **Retracted** v1.0 analytic formula after re-examination showed the product-form ansatz does not satisfy the full HJB first-order condition for $r > 0$ |
| v1.2 | Apr 2026 | Bayesian GZ rule ($r = 0$) established as rigorous result; non-separability for $r > 0$ proved; κ̄ formula corrected to reference posterior mean $\bar\mu$ |
| v1.3 | May 2026 | Stress tests added (5 return environments); figure placement fixed; editorial revision (claim softening, theorem → proposition for BGZ) |

---

## Related Papers

This paper is the third in a series:

1. **Sukhov (2026a)** — *Dynamic De-Risking under Drawdown Constraints*  
   Single-state drawdown HJB; Exponential DDR heuristic; $\lambda^* = 0.89$

2. **Sukhov (2026b)** — *Bayesian Kelly Criterion with Parameter Uncertainty*  
   Bayesian position sizing without drawdown barrier; Calmar ratio analysis

3. **Sukhov (2026c)** — *This paper*  
   Joint 2D HJB in $(d_t, n_t)$; Bayes GZ rule; stress tests

---

## Citation

```bibtex
@techreport{Sukhov2026c,
  author      = {Sukhov, Sergei},
  title       = {The {Bayesian Grossman-Zhou} Rule:
                 Drawdown-Constrained {Kelly} Betting
                 with Parameter Uncertainty},
  institution = {Market Microstructure Research Lab},
  year        = {2026},
  type        = {Working Paper},
  note        = {Revised May 2026},
  url         = {https://mmrls.com}
}
```

---

## Contact

**Sergei Sukhov**  
Market Microstructure Research Lab  
[research@mmrls.com](mailto:research@mmrls.com) · [mmrls.com](https://mmrls.com)

---

*MIT licence for code. Paper rights reserved.*
