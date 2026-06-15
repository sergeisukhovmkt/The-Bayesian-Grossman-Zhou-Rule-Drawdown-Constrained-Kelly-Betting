# Changelog

All notable changes to this project are documented here.

## [1.3.0] — May 2026 · Current

### Added
- **Section 9: Stress Tests** across five return environments:
  - A. GBM baseline (replication)
  - B. Markov regime switching (2-state hidden Markov, p_switch = 0.02/yr)
  - C. Student-t fat tails (ν = 4 degrees of freedom)
  - D. Jump diffusion (Merton 1976, λ = 2 jumps/yr, μ_J = −5%)
  - E. GARCH(1,1) volatility clustering (α = 0.10, β = 0.85)
- **Figure 7**: four-panel stress test comparison (utility, survival, ΔU, ΔSurv)
- `code/stress_tests.py`: vectorised simulation for all five environments
- `code/fig7_stress_tests.py`: Figure 7 generation script
- `Makefile` with targets: `fast`, `simulate`, `stress`, `figures`, `paper`, `reproduce`
- `requirements.txt`
- `CHANGELOG.md`

### Changed (editorial revision per ТЗ)
- **C1** Removed all instances of "exact closed-form solution" and
  "rigorous analytic rule"; replaced with "tractable stationary benchmark"
  and "candidate leverage policy"
- **C2** `\begin{theorem}[Bayes GZ]` → `\begin{proposition}[Stationary benchmark]`;
  all 8 cross-references updated; Appendix proof header updated
- **C3** Asymptotic bound "≈12.5% uniformly bounded" → "expected to be of order
  O(r/μ) — numerically small at baseline parameters"
- Abstract rewritten to 195 words (was ~300); problem → method → result structure
- Section "Economic Intuition" added between Introduction and Model Setup
- Robustness section updated with corrected simulation numbers

### Fixed
- Figure placement: all 7 figures now appear in the correct section
  (`[htbp]` + `\FloatBarrier` + `\clearpage` before wide figures)
- Fig 2 (value function) and Fig 4 (crossover analysis) were missing from
  `paper.tex` — both added with correct captions and labels
- `resizebox` applied to 8-column stress test table (overfull hbox resolved)
- `\emergencystretch = 3em` and `microtype` spacing options added globally
- `kbar()` function accepts numpy arrays (was scalar-only)

---

## [1.2.0] — April 2026

### Changed (mathematical corrections)
- **κ̄ formula corrected**: `κ̄(n) = (n₀μ₀ + nμ)/((n₀+n)σ²)` replaced with
  `κ̄_t = (μ̄_t − r)/σ²` where `μ̄_t` is the posterior mean; true μ is
  unobservable and must not appear in the policy formula
- **Assumption 2.2 added**: pre-trade estimation regime formalises when the
  2D reduction V(d,n) is valid (μ̄ fixed during trading horizon)
- **Remark added**: near-barrier asymptotics from the full 3D HJB independently
  confirm Bayes GZ (f* ≈ κ̄d/b as d → 0, V ~ ln(d))
- All anonymous reviewer citations removed from bibliography and text;
  revision history phrased neutrally

### Added
- `Assumption 2.2` (pre-trade estimation regime)
- Near-barrier remark (Remark following Proposition 1)
- 4 new bibliography entries: Garlappi et al. (2007), Pástor & Veronesi (2009),
  Xia (2001), Cover & Ordentlich (1996)

---

## [1.1.0] — April 2026

### Changed (retraction of v1.0 result)
- **Retracted**: analytic policy `f*(d,n) = κ̄(n)d/(1−α(n))` for r > 0
  was claimed in v1.0 based on a product-form ansatz
  `V(d,n) = A(n)(d/b)^α(n)`. A re-examination showed the value-of-learning
  term `f²V_n/σ²` introduces a d² nonlinearity in the HJB first-order
  condition, which no separable policy `g(n)d` can satisfy for all d ∈ (0,b]
- **Theorem 2 added** (Non-separability): proves that for r > 0 no separable
  linear policy satisfies the full HJB FOC
- **Theorem 3 added** (Asymptotic expansion): Bayes GZ is the leading-order
  term in the r/μ expansion; O(r/μ) correction requires numerical HJB solve
- Main theoretical result recentred on **Proposition 1 (Bayes GZ)** as the
  rigorously motivated benchmark for the undiscounted problem
- Simulation results unchanged (always tested Bayes GZ, not the retracted formula)

### Structural changes
- 10-block editorial revision applied:
  - Claims softened throughout ("tractable benchmark", "candidate policy")
  - Theorem count reduced: 5 → 2 Theorems + 3 Propositions + 2 Corollaries
  - Remark "Verification scope and limitations" added after main results
  - MC section: GBM-assumption caveat added
  - Closing paragraph rewritten (non-promotional)
  - References expanded (Garlappi, Pástor, Xia, Cover)
  - Author line: "Founder and Lead Researcher" removed

---

## [1.0.0] — April 2026 · Initial release

### Added
- Full 2D HJB formulation in state space (d_t, n_t)
- Kalman-Bucy derivation showing n_t is a controlled deterministic variable
- **(Retracted in v1.1)** Product-form solution V(d,n) = A(n)(d/b)^α(n) and
  analytic policy f*(d,n) = κ̄(n)d/(1−α(n)) for r > 0
- Double prudence theorem: ∂²f/∂d∂n > 0
- Convergence rate: D(n) ~ O(1/n), exact formula
- Monte Carlo validation: N = 50,000 GBM paths, 5 uncertainty scenarios
- Replication of Sukhov (2026a) results to within 0.3%
- Robustness: prior strength, TC sensitivity, rebalancing frequency
- 6 paper figures (PDF); `simulation.py`; `generate_figures.py`
