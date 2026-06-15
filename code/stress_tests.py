"""
stress_tests.py
────────────────────────────────────────────────────────────────────
Stress-test suite for the Bayesian Grossman-Zhou paper (Sukhov 2026c).

Three environments:
  A. Baseline GBM (constant drift, Gaussian)         — replication
  B. Markov-switching drift (2-state hidden Markov)  — regime instability
  C. Student-t returns (fat tails, ν=4 degrees)      — heavy tails
  D. Jump-diffusion (Merton 1976)                    — downside jumps
  E. GARCH-like clustered volatility                 — vol clustering

For each environment: Exp DDR oracle vs Exp DDR plug-in vs Bayes GZ.
N = 50,000 paths, seed = 42.
"""

import numpy as np
import json, time
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Model parameters (shared) ────────────────────────────────────────────
MU    = 0.08
SIG   = 0.20
DELTA = 0.20
T     = 2.0
DT    = 1/252
B     = -np.log(1 - DELTA)
KAP   = MU / SIG**2
N0    = 10.0
MU0   = 0.05
NSTEPS = int(T / DT)
PENALTY = -2.0


# ── Bayesian shrinkage ────────────────────────────────────────────────────
def kbar(n_obs_years, mu_hat):
    n = 12.0 * n_obs_years
    mu_post = (N0 * MU0 + n * mu_hat) / (N0 + n)
    return np.clip(mu_post, -0.3, 0.6) / SIG**2


# ── Leverage rules ────────────────────────────────────────────────────────
def f_oracle(d):
    """Exp DDR with true κ."""
    return KAP * (1 - np.exp(-0.89 * d / B))

def f_plugin(d, kap_hat):
    """Exp DDR with estimated κ̂."""
    return kap_hat * (1 - np.exp(-0.89 * d / B))

def f_bgz(d, n_obs_years, mu_hat):
    """Bayes GZ: κ̄(n) · d/b."""
    return kbar(n_obs_years, mu_hat) * d / B


# ── Core metrics ──────────────────────────────────────────────────────────
def metrics(flw: np.ndarray, surv: np.ndarray, rng, seed: int) -> dict:
    sr  = float(surv.mean())
    pU  = float(np.mean(flw * surv) + PENALTY * (1 - sr))
    ns  = int(surv.sum())
    ann = float(np.mean(flw[surv]) / T) if ns > 1 else float("nan")
    # Bootstrap CI
    rng2 = np.random.default_rng(seed + 77)
    boot = []
    N = len(flw)
    for _ in range(400):
        idx = rng2.integers(0, N, N)
        boot.append(np.mean(flw[idx]*surv[idx]) + PENALTY*(1-surv[idx].mean()))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return dict(pU=pU, sr=sr, ci=ci, ann=ann)


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT A: Baseline GBM
# ═══════════════════════════════════════════════════════════════════════════
def env_A(N: int, n_obs: float, sigma_u: float, seed: int) -> dict:
    """Standard GBM, constant drift."""
    rng = np.random.default_rng(seed)
    true_mu = np.clip(rng.normal(MU, sigma_u, N), -0.3, 0.5) if sigma_u > 0 \
              else np.full(N, MU)
    mu_hat  = np.clip(true_mu + rng.normal(0, SIG/np.sqrt(max(n_obs,0.01)), N), -0.3, 0.5)
    kap_hat = mu_hat / SIG**2

    results = {}
    for name, f_rule in [("oracle", lambda d, _1, _2: f_oracle(d)),
                          ("plugin", lambda d, kh, _:  f_plugin(d, kh)),
                          ("bgz",    lambda d, kh, mu: f_bgz(d, n_obs, mu))]:
        rng2 = np.random.default_rng(seed + 1)
        log_W = np.zeros(N); log_M = np.zeros(N); d = np.full(N, B)
        alive = np.ones(N, dtype=bool); final = np.zeros(N)
        for blk in range(0, NSTEPS, 63):
            end = min(blk+63, NSTEPS)
            shocks = rng2.standard_normal((end-blk, N)) * np.sqrt(DT)
            for s in range(end-blk):
                f = np.where(alive, np.clip(f_rule(d, kap_hat, mu_hat), 0, 4), 0)
                dX = (true_mu*f - 0.5*f**2*SIG**2)*DT + f*SIG*shocks[s]
                log_W += dX*alive; log_M = np.maximum(log_M, log_W)
                d = np.maximum(log_W - log_M + B, 0)
                ruin = alive & (d<=0)
                if ruin.any(): final[ruin]=log_W[ruin]; alive[ruin]=False
        final[alive] = log_W[alive]
        results[name] = metrics(final, alive, rng2, seed)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT B: Markov-switching drift
# ═══════════════════════════════════════════════════════════════════════════
def env_B(N: int, n_obs: float, sigma_u: float, seed: int,
          mu_bull: float = 0.12, mu_bear: float = -0.04,
          p_switch: float = 0.02) -> dict:
    """
    Two-state Markov regime: bull (μ=+12%) and bear (μ=-4%).
    Regime switches with probability p_switch per year.
    Investor does NOT observe regime; estimates μ from pre-trade data
    which may mix regimes.
    """
    rng = np.random.default_rng(seed)
    # Pre-trade μ̂: Gaussian noise around unconditional mean
    mu_unconditional = 0.5*(mu_bull + mu_bear)
    mu_hat = np.clip(
        mu_unconditional + rng.normal(0, SIG/np.sqrt(max(n_obs,0.01)), N),
        -0.4, 0.6
    )
    kap_hat = mu_hat / SIG**2

    # Simulate regime path for each path
    results = {}
    for name, f_rule in [("oracle", lambda d, kh, mu_t: f_oracle(d)),
                          ("plugin", lambda d, kh, mu_t: f_plugin(d, kh)),
                          ("bgz",    lambda d, kh, mu_t: f_bgz(d, n_obs, mu_hat))]:
        rng2 = np.random.default_rng(seed + 2)
        log_W = np.zeros(N); log_M = np.zeros(N); d = np.full(N, B)
        alive = np.ones(N, dtype=bool); final = np.zeros(N)
        # Initial regime: 50/50
        regime = rng2.integers(0, 2, N)  # 0=bull, 1=bear

        for blk in range(0, NSTEPS, 63):
            end = min(blk+63, NSTEPS)
            shocks = rng2.standard_normal((end-blk, N)) * np.sqrt(DT)
            for s in range(end-blk):
                # Regime switch
                switch = rng2.random(N) < p_switch * DT
                regime = np.where(switch, 1 - regime, regime)
                true_mu_t = np.where(regime==0, mu_bull, mu_bear)

                # Oracle uses unconditional mean (best a GBM trader can do)
                oracle_mu = mu_unconditional
                f_val = f_rule(d, kap_hat, mu_hat)
                if name == "oracle":
                    f_val = oracle_mu/SIG**2 * (1 - np.exp(-0.89*d/B))
                    f_val = np.clip(f_val, 0, 4)
                f = np.where(alive, np.clip(f_val, 0, 4), 0)

                dX = (true_mu_t*f - 0.5*f**2*SIG**2)*DT + f*SIG*shocks[s]
                log_W += dX*alive; log_M = np.maximum(log_M, log_W)
                d = np.maximum(log_W - log_M + B, 0)
                ruin = alive & (d<=0)
                if ruin.any(): final[ruin]=log_W[ruin]; alive[ruin]=False

        final[alive] = log_W[alive]
        results[name] = metrics(final, alive, rng2, seed)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT C: Student-t returns (fat tails, ν=4)
# ═══════════════════════════════════════════════════════════════════════════
def env_C(N: int, n_obs: float, sigma_u: float, seed: int, nu: float = 4.0) -> dict:
    """
    Fat-tailed returns: log-returns ~ t(ν) scaled to match GBM mean/var.
    ν=4: finite variance but heavy tails.
    """
    rng = np.random.default_rng(seed)
    true_mu = np.clip(rng.normal(MU, sigma_u, N), -0.3, 0.5) if sigma_u > 0 \
              else np.full(N, MU)
    mu_hat  = np.clip(true_mu + rng.normal(0, SIG/np.sqrt(max(n_obs,0.01)), N), -0.3, 0.5)
    kap_hat = mu_hat / SIG**2

    # Scale factor: t(ν) has variance ν/(ν-2), so σ_t = σ * sqrt((ν-2)/ν)
    t_scale = SIG * np.sqrt((nu - 2) / nu)

    results = {}
    for name, f_rule in [("oracle", lambda d, kh, mu: f_oracle(d)),
                          ("plugin", lambda d, kh, mu: f_plugin(d, kh)),
                          ("bgz",    lambda d, kh, mu: f_bgz(d, n_obs, mu))]:
        rng2 = np.random.default_rng(seed + 3)
        log_W = np.zeros(N); log_M = np.zeros(N); d = np.full(N, B)
        alive = np.ones(N, dtype=bool); final = np.zeros(N)

        for blk in range(0, NSTEPS, 63):
            end = min(blk+63, NSTEPS)
            # Student-t shocks (scaled)
            t_shocks = rng2.standard_t(nu, (end-blk, N)) * t_scale * np.sqrt(DT)
            for s in range(end-blk):
                f = np.where(alive, np.clip(f_rule(d, kap_hat, mu_hat), 0, 4), 0)
                dX = (true_mu*f - 0.5*f**2*SIG**2)*DT + f*t_shocks[s]
                log_W += dX*alive; log_M = np.maximum(log_M, log_W)
                d = np.maximum(log_W - log_M + B, 0)
                ruin = alive & (d<=0)
                if ruin.any(): final[ruin]=log_W[ruin]; alive[ruin]=False

        final[alive] = log_W[alive]
        results[name] = metrics(final, alive, rng2, seed)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT D: Jump diffusion (Merton 1976)
# ═══════════════════════════════════════════════════════════════════════════
def env_D(N: int, n_obs: float, sigma_u: float, seed: int,
          lam_jump: float = 2.0,    # jumps per year
          mu_jump:  float = -0.05,  # mean log-jump
          sig_jump: float = 0.08    # std of log-jump
         ) -> dict:
    """
    Merton jump diffusion: additional downward Poisson jumps.
    lam_jump=2: on average 2 jumps/year.
    mu_jump=-5%: mean log-jump (negative = downward pressure).
    """
    rng = np.random.default_rng(seed)
    true_mu = np.clip(rng.normal(MU, sigma_u, N), -0.3, 0.5) if sigma_u > 0 \
              else np.full(N, MU)
    mu_hat  = np.clip(true_mu + rng.normal(0, SIG/np.sqrt(max(n_obs,0.01)), N), -0.3, 0.5)
    kap_hat = mu_hat / SIG**2

    results = {}
    for name, f_rule in [("oracle", lambda d, kh, mu: f_oracle(d)),
                          ("plugin", lambda d, kh, mu: f_plugin(d, kh)),
                          ("bgz",    lambda d, kh, mu: f_bgz(d, n_obs, mu))]:
        rng2 = np.random.default_rng(seed + 4)
        log_W = np.zeros(N); log_M = np.zeros(N); d = np.full(N, B)
        alive = np.ones(N, dtype=bool); final = np.zeros(N)

        for blk in range(0, NSTEPS, 63):
            end = min(blk+63, NSTEPS)
            gauss = rng2.standard_normal((end-blk, N)) * np.sqrt(DT)
            # Poisson jump arrivals
            n_jumps = rng2.poisson(lam_jump * DT, (end-blk, N))
            jump_sizes = rng2.normal(mu_jump, sig_jump, (end-blk, N))

            for s in range(end-blk):
                f = np.where(alive, np.clip(f_rule(d, kap_hat, mu_hat), 0, 4), 0)
                # Diffusion component
                dX_diff = (true_mu*f - 0.5*f**2*SIG**2)*DT + f*SIG*gauss[s]
                # Jump component: f * sum(log-jumps)
                dX_jump = f * n_jumps[s] * jump_sizes[s]
                dX = dX_diff + dX_jump
                log_W += dX*alive; log_M = np.maximum(log_M, log_W)
                d = np.maximum(log_W - log_M + B, 0)
                ruin = alive & (d<=0)
                if ruin.any(): final[ruin]=log_W[ruin]; alive[ruin]=False

        final[alive] = log_W[alive]
        results[name] = metrics(final, alive, rng2, seed)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT E: GARCH(1,1) volatility clustering
# ═══════════════════════════════════════════════════════════════════════════
def env_E(N: int, n_obs: float, sigma_u: float, seed: int,
          omega: float = 0.00001,
          alpha: float = 0.10,
          beta:  float = 0.85) -> dict:
    """
    GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Unconditional variance = ω/(1-α-β). Calibrated to SIG=0.20 unconditionally.
    """
    rng = np.random.default_rng(seed)
    # Calibrate omega to match unconditional σ=0.20 daily
    sig_daily = SIG * np.sqrt(DT)
    omega_cal = sig_daily**2 * (1 - alpha - beta)

    true_mu = np.clip(rng.normal(MU, sigma_u, N), -0.3, 0.5) if sigma_u > 0 \
              else np.full(N, MU)
    mu_hat  = np.clip(true_mu + rng.normal(0, SIG/np.sqrt(max(n_obs,0.01)), N), -0.3, 0.5)
    kap_hat = mu_hat / SIG**2

    results = {}
    for name, f_rule in [("oracle", lambda d, kh, mu: f_oracle(d)),
                          ("plugin", lambda d, kh, mu: f_plugin(d, kh)),
                          ("bgz",    lambda d, kh, mu: f_bgz(d, n_obs, mu))]:
        rng2 = np.random.default_rng(seed + 5)
        log_W = np.zeros(N); log_M = np.zeros(N); d = np.full(N, B)
        alive = np.ones(N, dtype=bool); final = np.zeros(N)
        sig2 = np.full(N, sig_daily**2)   # conditional variance state

        z_all = rng2.standard_normal((NSTEPS, N))

        for t in range(NSTEPS):
            # Current conditional σ
            sig_t = np.sqrt(sig2)
            f = np.where(alive, np.clip(f_rule(d, kap_hat, mu_hat), 0, 4), 0)
            eps = sig_t * z_all[t]
            dX  = true_mu*f*DT + f*eps
            log_W += dX*alive; log_M = np.maximum(log_M, log_W)
            d = np.maximum(log_W - log_M + B, 0)
            ruin = alive & (d<=0)
            if ruin.any(): final[ruin]=log_W[ruin]; alive[ruin]=False
            # Update GARCH variance
            sig2 = omega_cal + alpha * (f*eps)**2 + beta * sig2
            sig2 = np.clip(sig2, 1e-8, 0.5**2)

        final[alive] = log_W[alive]
        results[name] = metrics(final, alive, rng2, seed)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: run all environments
# ═══════════════════════════════════════════════════════════════════════════
def run_all(N: int = 50_000, seed: int = 42):
    n_obs   = 2.0       # cold-start: 2 years pre-trade data
    sigma_u = 0.03

    print(f"N={N:,} paths  ·  n_obs={n_obs}yr  ·  σ_u={sigma_u}")
    print(f"{'Environment':32s} {'Strategy':10s} {'U(π)':>8} {'CI':>18} {'Surv%':>7} {'Ann%':>7}")
    print("─" * 88)

    envs = [
        ("A. GBM baseline (Gaussian)",     env_A),
        ("B. Markov regime switching",      env_B),
        ("C. Student-t fat tails (ν=4)",   env_C),
        ("D. Jump diffusion (Merton)",      env_D),
        ("E. GARCH vol clustering",         env_E),
    ]

    all_results = {}
    for label, fn in envs:
        t0 = time.time()
        res = fn(N, n_obs, sigma_u, seed)
        dt = time.time() - t0
        all_results[label] = res
        printed = False
        for strat in ["oracle","plugin","bgz"]:
            r = res[strat]
            ann_s = f"{r['ann']*100:.2f}" if r['ann']==r['ann'] else " N/A"
            lbl   = label if not printed else ""
            mark  = " ◄" if strat=="bgz" else ""
            ci    = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
            print(f"  {lbl:30s} {strat:10s} {r['pU']:>+8.4f} {ci:>18} "
                  f"{r['sr']*100:>6.1f}% {ann_s:>7}{mark}")
            printed = True
        print(f"  {'':30s} {'':10s} {'':>8} {'':>18} {'':>7} {'':>7}  [{dt:.1f}s]")

    # ── Double-prudence check across environments ──────────────────────────
    print()
    print("─" * 55)
    print("Double prudence: ΔU(BGZ−plug-in) by environment")
    print("─" * 55)
    for label, _ in envs:
        r_bgz  = all_results[label]["bgz"]
        r_plug = all_results[label]["plugin"]
        delta  = r_bgz["pU"] - r_plug["pU"]
        delta_surv = (r_bgz["sr"] - r_plug["sr"]) * 100
        print(f"  {label:35s}  ΔU={delta:>+7.4f}  ΔSurv={delta_surv:>+5.1f}pp")

    # Save for figure generation
    with open("/tmp/stress_results.json","w") as f:
        safe = {k: {s: {m: (v if m!="ci" else list(v))
                        for m,v in res.items() if m not in ("ann",) or True}
                    for s,res in v2.items()}
                for k,v2 in all_results.items()}
        json.dump(safe, f, indent=2, default=str)
    print("\nSaved /tmp/stress_results.json")
    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--N", type=int, default=50_000)
    args = p.parse_args()
    N = 5_000 if args.fast else args.N
    run_all(N=N)
