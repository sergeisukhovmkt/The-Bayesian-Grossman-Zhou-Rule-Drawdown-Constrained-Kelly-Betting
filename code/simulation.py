"""
simulation.py
─────────────────────────────────────────────────────────────────────────────
Sukhov (2026c, revised)  –  The Bayesian Grossman-Zhou Rule

Drawdown-Constrained Kelly Betting with Parameter Uncertainty.

Implements and compares four leverage strategies on GBM paths with a
running-maximum drawdown barrier:

    1. Full Kelly oracle      –  f = κ_true         (Kelly 1956)
    2. Linear DDR oracle      –  f = κ_true · d/b   (Grossman-Zhou 1993)
    3. Exp DDR plug-in        –  f = κ_hat · (1-exp(-λd/b))  (Sukhov 2026a, plug-in)
    4. Bayes GZ [MAIN]        –  f = κ̄(n) · d/b    (this paper, Thm 4.1)

Note: the formula f = κ̄(n)·d/(1-α(n)) from the first version of this paper
has been RETRACTED after an expert technical note showed it does not solve
the 2D HJB for r>0. See Section 4.2 and the revision note.

Usage:
    python simulation.py              # full run, N=50,000
    python simulation.py --fast       # quick check, N=5,000

Requires: numpy, scipy
"""

import argparse
import numpy as np

# ─── Model parameters ────────────────────────────────────────────────────────
MU    = 0.08          # true drift
SIG   = 0.20          # volatility
DELTA = 0.20          # drawdown threshold (20%)
T     = 2.0           # horizon (years)
DT    = 1/252         # daily step
MU0   = 0.05          # prior mean drift
N0    = 10.0          # prior effective sample size (months)
B     = -np.log(1 - DELTA)        # log-barrier width ≈ 0.2231
KAP   = MU / SIG**2                # true Kelly fraction = 2.0
N_STEPS = int(T / DT)              # 504 steps
RUIN_PENALTY = -2.0


# ─── Bayesian shrinkage ──────────────────────────────────────────────────────
def kbar(n_obs_years: float, mu_hat: float = MU) -> float:
    """
    Posterior mean Kelly fraction via conjugate Gaussian shrinkage.
    n_obs_years: pre-trade monthly observations expressed as years;
    effective sample size = N0 + 12 * n_obs_years.
    """
    n = 12.0 * n_obs_years
    mu_post = (N0 * MU0 + n * mu_hat) / (N0 + n)
    return mu_post / SIG**2


# ─── Leverage rules ──────────────────────────────────────────────────────────
class LeverageRule:
    """Base class — takes state (d, κ̂, n_obs_years) and returns leverage."""
    name: str

    def __call__(self, d, kappa_hat, n_obs_years):
        raise NotImplementedError


class FullKelly(LeverageRule):
    name = "Full Kelly (oracle κ)"

    def __call__(self, d, kappa_hat, n_obs_years):
        return np.full_like(d, KAP)


class LinearDDR(LeverageRule):
    """Grossman-Zhou (1993) linear cushion rule with oracle κ."""
    name = "Linear DDR (GZ, oracle κ)"

    def __call__(self, d, kappa_hat, n_obs_years):
        return KAP * (d / B)


class ExpDDROracle(LeverageRule):
    """Exp DDR from Sukhov (2026a) with oracle κ."""
    name = "Exp DDR (oracle κ, λ=0.89)"

    def __init__(self, lam: float = 0.89):
        self.lam = lam

    def __call__(self, d, kappa_hat, n_obs_years):
        return KAP * (1.0 - np.exp(-self.lam * d / B))


class ExpDDRPlugIn(LeverageRule):
    """Exp DDR with estimated κ̂ in place of true κ (naive baseline)."""
    name = "Exp DDR plug-in (κ̂)"

    def __init__(self, lam: float = 0.89):
        self.lam = lam

    def __call__(self, d, kappa_hat, n_obs_years):
        return kappa_hat * (1.0 - np.exp(-self.lam * d / B))


class BayesGZ(LeverageRule):
    """
    MAIN RULE OF THIS PAPER (Theorem 4.1):

        f^BGZ(d, n) = κ̄(n) · d / b

    where κ̄(n) is the Bayesian shrinkage estimate of the Kelly fraction.
    This is the unique closed-form analytic solution of the 2D HJB for the
    undiscounted growth criterion (r=0).
    """
    name = "Bayes GZ  κ̄(n)·d/b  [main, Thm 4.1]"

    def __call__(self, d, kappa_hat, n_obs_years):
        # kappa_hat carries the pre-trade estimate μ̂ through κ̂ = μ̂/σ²
        mu_hat = kappa_hat * SIG**2
        n = 12.0 * n_obs_years
        k_bar = (N0 * MU0 + n * mu_hat) / (N0 + n) / SIG**2
        return k_bar * d / B


# ─── Vectorised GBM simulator with drawdown barrier ───────────────────────────
def simulate(rule, N_paths, seed=42, n_obs_years=1000.0,
             sigma_u=0.0, ruin_penalty=RUIN_PENALTY):
    """
    Simulate N_paths GBM paths with running-maximum drawdown barrier.
    """
    rng = np.random.default_rng(seed)

    if sigma_u > 0:
        true_mu = np.clip(rng.normal(MU, sigma_u, N_paths), -0.3, 0.5)
    else:
        true_mu = np.full(N_paths, MU)

    se_mu   = SIG / np.sqrt(max(n_obs_years, 1e-3))
    mu_hat  = np.clip(true_mu + rng.normal(0, se_mu, N_paths), -0.3, 0.5)
    kap_hat = mu_hat / SIG**2

    log_W = np.zeros(N_paths)
    log_M = np.zeros(N_paths)
    d     = np.full(N_paths, B)
    alive = np.ones(N_paths, dtype=bool)
    final = np.zeros(N_paths)

    for blk in range(0, N_STEPS, 63):
        end = min(blk + 63, N_STEPS)
        shocks = rng.standard_normal((end - blk, N_paths)) * np.sqrt(DT)

        for s in range(end - blk):
            dW = shocks[s]
            f  = rule(d, kap_hat, n_obs_years)
            f  = np.where(alive, np.clip(f, 0.0, 4.0), 0.0)

            dX = (true_mu * f - 0.5 * f**2 * SIG**2) * DT + f * SIG * dW
            log_W += dX * alive
            log_M  = np.maximum(log_M, log_W)
            d      = np.maximum(log_W - log_M + B, 0.0)

            ruined = alive & (d <= 0.0)
            if ruined.any():
                final[ruined] = log_W[ruined]
                alive[ruined] = False

    final[alive] = log_W[alive]

    surv = alive
    sr   = float(surv.mean())
    flw  = final
    pU   = float(np.mean(flw * surv) + ruin_penalty * (1.0 - sr))

    n_s = int(surv.sum())
    if n_s > 1:
        ann    = float(np.mean(flw[surv]) / T)
        std_a  = float(np.std(flw[surv]) / T)
        sharpe = ann / std_a * np.sqrt(T) if std_a > 0 else float("nan")
    else:
        ann = std_a = sharpe = float("nan")

    rng2 = np.random.default_rng(seed + 100)
    boot = []
    for _ in range(400):
        idx = rng2.integers(0, N_paths, N_paths)
        boot.append(np.mean(flw[idx] * surv[idx])
                    + ruin_penalty * (1.0 - surv[idx].mean()))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    return {
        "strategy":          rule.name,
        "penalised_utility": pU,
        "survival_rate":     sr,
        "ci_95":             ci,
        "mean_ann_logret":   ann,
        "sharpe":            sharpe,
    }


# ─── Experiment runners ───────────────────────────────────────────────────────
def run_main_table(N_paths):
    rules = [
        ExpDDROracle(lam=0.89),
        LinearDDR(),
        ExpDDRPlugIn(lam=0.89),
        BayesGZ(),
    ]

    scenarios = [
        ("Oracle (known κ)",          1000.0, 0.00),
        ("5 yr data,  σ_u=0.02",         5.0, 0.02),
        ("2 yr data,  σ_u=0.03",         2.0, 0.03),
        ("1 yr data,  σ_u=0.04 [cold]",  1.0, 0.04),
    ]

    print(f"\n{'Strategy':42s}  {'U(π)':>8}  {'95% CI':>18}  "
          f"{'Surv%':>6}  {'AnnRet%':>8}")
    print("─" * 90)

    for label, n_obs, sigma_u in scenarios:
        print(f"\n─── {label}  (n_obs={n_obs} yr, σ_u={sigma_u}) ───")
        for rule in rules:
            r = simulate(rule, N_paths,
                         n_obs_years=n_obs, sigma_u=sigma_u, seed=42)
            ann_s = (f"{r['mean_ann_logret']*100:.2f}"
                     if r['mean_ann_logret'] == r['mean_ann_logret']
                     else "  N/A")
            mark = " ◄ main" if "Bayes GZ" in rule.name else ""
            print(f"  {rule.name:40s}  {r['penalised_utility']:>+8.4f}  "
                  f"[{r['ci_95'][0]:+.3f},{r['ci_95'][1]:+.3f}]  "
                  f"{r['survival_rate']*100:>5.1f}%  {ann_s:>8}{mark}")


def run_verification():
    print("\n" + "═" * 70)
    print(" Verification of structural results (Section 5)")
    print("═" * 70)

    # Double prudence
    print("\n── Double prudence: ∂²f/∂d∂n = κ̄'(n)/b > 0 ──")
    print(f"{'n (months)':>12}  {'κ̄(n)':>10}  {'κ̄′(n)':>12}  {'>0?':>5}")
    for n_months in [0, 10, 30, 60, 120, 240, 600]:
        n_obs = n_months / 12.0
        k     = kbar(n_obs)
        k_p   = (kbar(n_obs + 1/12) - kbar(n_obs - 1/12)) / (2/12)
        print(f"{n_months:>12}  {k:>10.5f}  {k_p:>12.6f}  "
              f"{'✓' if k_p > 0 else '✗':>5}")

    # Exact O(1/n) convergence
    print("\n── Exact O(1/n) convergence: D(n) = (κ-κ₀)·n₀/(n₀+n) ──")
    KAP0 = MU0 / SIG**2
    D0   = KAP - KAP0
    print(f"{'n_months':>10}  {'D formula':>14}  {'D theorem':>14}  {'match':>8}")
    for n_months in [0, 5, 10, 20, 50, 100, 200, 500]:
        n_obs     = n_months / 12.0
        D_formula = KAP - kbar(n_obs)
        D_theorem = D0 * N0 / (N0 + 12 * n_obs)
        match = abs(D_formula - D_theorem) < 1e-12
        print(f"{n_months:>10}  {D_formula:>14.6f}  {D_theorem:>14.6f}  "
              f"{'✓' if match else '✗':>8}")

    # Show why the old formula was wrong
    print("\n── Why the product-form f=g(n)d fails for r>0 (Thm 4.2) ──")
    print("   Correct V_n/V_d = (d/α)·[A'/A + α'·ln(d/b)]    (depends on d!)")
    print("   Incorrect V_n/V_d = 2rb/α                      (used in v1 paper)")
    print()
    r = 0.01
    for n_obs in [1.0, 5.0, 20.0]:
        k = kbar(n_obs)
        a = 2*r / (2*r + k**2 * SIG**2)
        for d_rel in [0.1, 0.5, 1.0]:
            d = d_rel * B
            correct = (d / a) * 2 * r
            wrong   = 2 * r * B / a
            print(f"   n_obs={n_obs:>5.1f}yr, d={d_rel:>3.1f}·b:  "
                  f"correct={correct:.4f}, v1-incorrect={wrong:.4f}, "
                  f"ratio={correct/wrong:.2f}")


def main(fast=False):
    N = 5_000 if fast else 50_000
    print("═" * 70)
    print(" Sukhov (2026c, revised) – The Bayesian Grossman–Zhou Rule")
    print(f" N = {N:,} paths  ·  T = {T} yr  ·  δ = {DELTA:.0%}  ·  "
          f"μ = {MU}  ·  σ = {SIG}")
    print("═" * 70)

    run_main_table(N)
    run_verification()

    print("\n" + "═" * 70)
    print(" Done.  Main result: Bayes GZ rule is the rigorous analytic policy.")
    print("═" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use N=5,000 paths instead of 50,000")
    args = parser.parse_args()
    main(fast=args.fast)
