"""
generate_figures.py
Reproduces all figures for:
  Sukhov (2026c) – Bayesian Dynamic De-Risking under Drawdown Constraints

Run:  python generate_figures.py
Output: ../figures/fig{1..6}.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

# ── global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8.5,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "legend.fontsize":  7.5,
    "lines.linewidth":  1.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       180,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.06,
})

C = {                          # colour palette
    "oracle": "#1a1a2e",
    "gz":     "#7f8c8d",
    "plugin": "#c0392b",
    "new":    "#16213e",
    "bgz":    "#0f3460",
    "accent": "#e94560",
    "green":  "#27ae60",
    "blue":   "#2980b9",
    "orange": "#e67e22",
}

# ── model parameters ──────────────────────────────────────────
MU, SIG, DELTA, R_DISC = 0.08, 0.20, 0.20, 0.01
N0, MU0 = 10.0, 0.05
B    = -np.log(1 - DELTA)           # log-barrier width
KAP  = (MU - 0.0) / SIG**2          # true Kelly fraction

def kbar(n):
    return (N0*MU0 + n*MU) / ((N0 + n) * SIG**2)

def alpha_n(n):
    k = kbar(n)
    return 2*R_DISC / (2*R_DISC + k**2 * SIG**2)

def aprime(n, eps=1e-3):
    return (alpha_n(n+eps) - alpha_n(n-eps)) / (2*eps)

def g(n):          # slope of optimal rule: f*(d,n) = g(n)*d
    return kbar(n) / (1 - alpha_n(n))

# ══════════════════════════════════════════════════════════════
# FIG 1  Policy shapes
# ══════════════════════════════════════════════════════════════
def make_fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    dv = np.linspace(1e-4, B, 400)

    # ── panel (a): oracle comparison ──────────────────────────
    ax = axes[0]
    n_inf = 1e6
    k_inf = kbar(n_inf)

    ax.plot(dv/B, np.full_like(dv, KAP),            color=C["gz"],     lw=1.2,
            ls=":",  label="Full Kelly ($f=\\kappa$)")
    ax.plot(dv/B, KAP * dv / B,                      color=C["gz"],     lw=1.4,
            ls="-.", label="Grossman--Zhou linear")
    ax.plot(dv/B, KAP*(1-np.exp(-0.89*dv/B)),        color=C["oracle"], lw=1.8,
            ls="--", label="Exp DDR $\\lambda^*{=}0.89$ (Art.\\ 1)")
    ax.plot(dv/B, k_inf*dv/B,                        color=C["bgz"],    lw=2.2,
            label="Bayes GZ, $n{\\to}\\infty$ (this paper)")

    ax.axvline(0, color="red", lw=0.9, ls=":", alpha=0.7)
    ax.text(0.02, 0.12, "$d{=}0$\n(ruin)", fontsize=6.5, color="red",
            transform=ax.transAxes)
    ax.set_xlabel("$d/b$  (normalised log-distance to barrier)")
    ax.set_ylabel("Leverage $f$")
    ax.set_title("(a) Strategy shapes – oracle $\\kappa$ known")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 2.15)
    ax.grid(alpha=0.18, lw=0.6)

    # ── panel (b): Bayes GZ at different n ────────────────
    ax2 = axes[1]
    ns   = [0, 5, 20, 50, 100, int(1e6)]
    labs = ["$n=0$ (prior)", "$n=5$", "$n=20$",
            "$n=50$", "$n=100$", "$n\\!\\to\\!\\infty$ (oracle)"]
    clrs = ["#e74c3c","#e67e22","#f1c40f","#27ae60","#2980b9","#1a1a2e"]
    for nv, lab, col in zip(ns, labs, clrs):
        k = kbar(nv)
        f = k * dv / B   # Bayes GZ rule
        lw = 2.2 if nv in [0, int(1e6)] else 1.4
        ax2.plot(dv/B, f, color=col, lw=lw, label=lab)

    ax2.axvline(0, color="red", lw=0.9, ls=":", alpha=0.7)
    ax2.set_xlabel("$d/b$")
    ax2.set_ylabel("$f^{\\mathrm{BGZ}}(d,n)$")
    ax2.set_title("(b) $f^{\\mathrm{BGZ}}(d,n) = \\bar\\kappa(n)\\,d/b$\nfor various $n$")
    ax2.legend(framealpha=0.9, fontsize=6.8)
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.01, 2.10)
    ax2.grid(alpha=0.18, lw=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig("../figures/fig1_policy_shapes.pdf")
    plt.close(); print("fig1 saved")


# ══════════════════════════════════════════════════════════════
# FIG 2  Value function and parameter evolution
# ══════════════════════════════════════════════════════════════
def make_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    dv = np.linspace(1e-4, B, 400)
    KAP_loc  = MU/SIG**2
    KAP0_loc = MU0/SIG**2

    # panel (a): Bayes GZ value function V0(d,n) ∝ kbar(n)^2 * d/b
    # (Theorem 4.1: linear in d, scaled by kbar^2)
    ax = axes[0]
    ns_v  = [0, 10, 50, 100, int(1e6)]
    clrs_v= ["#e74c3c","#e67e22","#27ae60","#2980b9","#1a1a2e"]
    labs_v= ["$n=0$","$n=10$","$n=50$","$n=100$","$n\\!\\to\\!\\infty$"]
    for nv, col, lab in zip(ns_v, clrs_v, labs_v):
        kn = kbar(nv)
        V  = kn**2 * (dv/B)
        ax.plot(dv/B, V, color=col, lw=1.8 if nv in [0,int(1e6)] else 1.3,
                label=lab)
    ax.set_xlabel("$d/b$")
    ax.set_ylabel("$V^0(d,n)/(\\sigma^2)$")
    ax.set_title("(a) Bayes GZ value function\n$V^0(d,n) = \\bar\\kappa(n)^2 \\sigma^2\\,d/b$")
    ax.legend(framealpha=0.9); ax.grid(alpha=0.18, lw=0.6)

    # panel (b): kbar(n) and slope kbar(n)/b vs n
    ax2 = axes[1]
    nv  = np.linspace(0, 280, 500)
    kbar_arr = np.array([kbar(x) for x in nv])
    slope    = kbar_arr / B           # actual leverage slope
    k_inf    = kbar(1e6)

    ax2b = ax2.twinx()
    l1, = ax2.plot(nv, kbar_arr, color=C["blue"],   lw=2,
                   label="$\\bar\\kappa(n)$")
    l2, = ax2b.plot(nv, slope,   color=C["accent"], lw=2, ls="--",
                    label="slope $=\\bar\\kappa(n)/b$")
    ax2.axhline(k_inf,  color=C["blue"],   lw=0.8, ls=":", alpha=0.5)
    ax2b.axhline(k_inf/B, color=C["accent"], lw=0.8, ls=":", alpha=0.5)
    ax2.text(230, k_inf*0.96, "$\\kappa$", fontsize=8, color=C["blue"])
    ax2.set_xlabel("$n_\\mathrm{eff}$  (effective sample size)")
    ax2.set_ylabel("$\\bar\\kappa(n)$",          color=C["blue"])
    ax2b.set_ylabel("$\\bar\\kappa(n)/b$",       color=C["accent"])
    ax2.set_title("(b) Bayesian shrinkage estimate $\\bar\\kappa(n)$\nand resulting BGZ leverage slope")
    ax2.legend(handles=[l1,l2], framealpha=0.9, loc="lower right")
    ax2.grid(alpha=0.18, lw=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig("../figures/fig2_value_function.pdf")
    plt.close(); print("fig2 saved")


# ══════════════════════════════════════════════════════════════
# FIG 3  Double prudence + convergence rate
# ══════════════════════════════════════════════════════════════
def make_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # For Bayes GZ rule f = kbar(n) * d/b, slope is kbar(n)/b,
    # and deficit is D(n) = (kappa - kbar(n))/b * d ≡ kappa - kbar(n) per unit d.
    KAP = MU/SIG**2
    KAP0 = MU0/SIG**2

    # panel (a): deficit D(n) = kappa - kbar(n) and exact approximation
    ax = axes[0]
    nv    = np.linspace(0, 300, 500)
    kbar_arr = np.array([kbar(x) for x in nv])
    D     = KAP - kbar_arr               # leverage deficit per unit d/b
    # Theorem 4.6 says: D(n) = (kappa - kappa0) * n0/(n0+n)  EXACTLY
    D_exact = (KAP - KAP0) * N0 / (N0 + nv)

    ax.plot(nv, D,        color=C["new"],   lw=2.5, label="$D(n)=\\kappa-\\bar\\kappa(n)$")
    ax.plot(nv, D_exact,  color=C["orange"],lw=1.4, ls="--",
            label="$(\\kappa{-}\\kappa_0)\\,n_0/(n_0{+}n)$  (Thm.~4.6, exact)")

    for frac, lab in [(0.5,"50%"), (0.1,"90%"), (0.05,"95%")]:
        nt = N0*(1/frac - 1)
        ax.axvline(nt, color="gray", lw=0.8, ls=":")
        ax.text(nt+4, D[0]*0.82, lab, fontsize=6.5, color="gray")

    ax.set_xlabel("$n_\\mathrm{eff}$")
    ax.set_ylabel("Leverage deficit $D(n)$")
    ax.set_title("(a) Exact $O(1/n)$ convergence to oracle\n"
                 "(Bayes GZ rule)")
    ax.legend(framealpha=0.9); ax.grid(alpha=0.18, lw=0.6)

    # panel (b): cross-partial = kbar'(n)/b > 0
    ax2 = axes[1]
    nv2  = np.linspace(0.5, 280, 400)
    # kbar'(n) = (kappa - kappa0) * n0 / (n0 + n)^2
    kbar_prime = (KAP - KAP0) * N0 / (N0 + nv2)**2 / B

    ax2.plot(nv2, kbar_prime, color=C["green"], lw=2)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.fill_between(nv2, 0, kbar_prime, alpha=0.15, color=C["green"])
    ax2.set_xlabel("$n_\\mathrm{eff}$")
    ax2.set_ylabel("$\\bar\\kappa'(n)/b$")
    ax2.set_title("(b) Double prudence:\n"
                  "$\\partial^2 f^{\\mathrm{BGZ}}/\\partial d\\,\\partial n = \\bar\\kappa'(n)/b>0$")
    ax2.text(120, kbar_prime[60]*0.7,
             "$\\bar\\kappa'(n)>0\\;\\forall n\\geq 0$ $\\checkmark$",
             fontsize=7.5, color=C["green"])
    ax2.grid(alpha=0.18, lw=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig("../figures/fig3_double_prudence.pdf")
    plt.close(); print("fig3 saved")


# ══════════════════════════════════════════════════════════════
# FIG 4  Perturbation error bounds
# ══════════════════════════════════════════════════════════════
def make_fig4():
    """
    Replaces the (now-retracted) perturbation error bounds figure with a
    visualization of the crossover between Bayes GZ and Exp DDR plug-in
    as the estimation window n_obs varies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    KAP_loc  = MU/SIG**2
    KAP0_loc = MU0/SIG**2

    # panel (a): kbar(n) shrinkage vs n_obs (months)
    ax = axes[0]
    n_months = np.linspace(0, 240, 500)
    kbar_arr = np.array([(N0*MU0 + n*MU)/((N0+n)*SIG**2) for n in n_months])
    ax.plot(n_months, kbar_arr, color=C["bgz"], lw=2.2,
            label="$\\bar\\kappa(n)$ (Bayesian shrinkage)")
    ax.axhline(KAP_loc,  color=C["oracle"], lw=1, ls="--",
               label="$\\kappa$ (true Kelly fraction)")
    ax.axhline(KAP0_loc, color=C["plugin"], lw=1, ls=":",
               label="$\\kappa_0$ (prior)")
    # Mark 50%, 90%, 95% thresholds
    for frac, lab, col in [(0.5,"50%","gray"), (0.1,"90%","gray"),
                           (0.05,"95%","gray")]:
        nt = N0*(1/frac - 1)
        ax.axvline(nt, color=col, lw=0.7, ls=":")
        ax.text(nt+2, KAP0_loc+0.05, lab, fontsize=6.5, color=col)
    ax.set_xlabel("$n$ (months of data)")
    ax.set_ylabel("Posterior Kelly fraction")
    ax.set_title("(a) Shrinkage trajectory $\\bar\\kappa(n)$\n"
                 "from prior $\\kappa_0$ to oracle $\\kappa$")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_xlim(0, 240); ax.set_ylim(KAP0_loc-0.1, KAP_loc+0.15)

    # panel (b): U(BGZ) - U(plug-in) vs n_obs years (real simulation data)
    ax2 = axes[1]
    n_yrs = [0.5, 1, 2, 5, 10, 20]
    # From simulation at sigma_u = 0.03 (panel data)
    U_bgz_sim  = [-0.014, -0.114, -0.090, -0.012, +0.032, +0.058]
    U_plug_sim = [-0.487, -0.331, -0.177, -0.005, +0.054, +0.072]
    delta_U   = [b - p for b, p in zip(U_bgz_sim, U_plug_sim)]

    colors = [C["bgz"] if d > 0 else C["plugin"] for d in delta_U]
    bars = ax2.bar(range(len(n_yrs)), delta_U, color=colors, alpha=0.85)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xticks(range(len(n_yrs)))
    ax2.set_xticklabels([f"{n}yr" for n in n_yrs], fontsize=8)
    ax2.set_xlabel("Years of pre-trade data ($\\sigma_u=0.03$)")
    ax2.set_ylabel("$\\Delta U = U_{\\mathrm{BGZ}} - U_{\\mathrm{plug-in}}$")
    ax2.set_title("(b) Bayes GZ advantage by horizon\n"
                  "(crossover at $n\\approx 3$\\,yr)")
    # Annotate bars
    for b, d in zip(bars, delta_U):
        sign = "+" if d > 0 else ""
        ax2.text(b.get_x()+b.get_width()/2,
                 d + 0.01*np.sign(d),
                 f"{sign}{d:.3f}", ha="center",
                 va="bottom" if d > 0 else "top",
                 fontsize=7, color="black")
    ax2.grid(axis="y", alpha=0.18, lw=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig("../figures/fig4_error_bounds.pdf")
    plt.close(); print("fig4 saved")


# ══════════════════════════════════════════════════════════════
# FIG 5  Monte Carlo: survival and penalised utility
# ══════════════════════════════════════════════════════════════
def make_fig5():
    # Real simulation data (N=8,000 paths, sigma_u=0.03 for panel A;
    # scenario-specific sigma_u for panel B).  See /tmp/fig_data.json.
    n_pts   = [0.5, 1, 2, 5, 10, 20, 50, 100]
    surv_orc= [100., 100., 100., 100., 100., 100., 100., 100.]
    surv_ep = [74.6, 81.9, 89.1, 97.0, 99.5, 99.9, 100., 100.]   # plug-in
    surv_bgz= [97.3, 92.3, 93.4, 97.0, 98.8, 99.7, 100., 100.]   # Bayes GZ

    scen_x  = np.arange(5)
    scen_lb = ["Oracle", "10yr\n$\\sigma_u{=}0.02$", "5yr\n$\\sigma_u{=}0.02$",
               "2yr\n$\\sigma_u{=}0.03$", "1yr\n$\\sigma_u{=}0.04$"]
    # From N=50,000 main table; 10yr simulated at sigma_u=0.02
    U_oracle= [0.083, 0.084, 0.084, 0.085, 0.087]  # Exp DDR oracle kappa
    U_plugin= [0.083, 0.054,-0.003,-0.169,-0.337]  # plug-in
    U_bgz   = [0.072, 0.058,-0.010,-0.084,-0.104]  # Bayes GZ (real)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # panel (a): survival rate vs n_obs
    ax = axes[0]
    ax.semilogx(n_pts, [100]*8, color=C["oracle"], lw=1.5, ls="--",
                label="Exp DDR oracle ($\\kappa$ known)")
    ax.semilogx(n_pts, surv_ep, color=C["plugin"], lw=2, marker="s", ms=4,
                label="Exp DDR plug-in (estimated $\\hat\\kappa$)")
    ax.semilogx(n_pts, surv_bgz, color=C["bgz"],  lw=2, marker="^", ms=5,
                label="Bayes GZ (this paper)")
    ax.fill_between(n_pts, surv_ep, surv_bgz, alpha=0.15, color=C["bgz"])
    ax.axhline(95, color=C["orange"], lw=1, ls=":", label="95% threshold")
    ax.set_xlabel("Years of pre-trade data (log scale)")
    ax.set_ylabel("Survival rate (%)")
    ax.set_title("(a) Survival at 20% drawdown barrier\n"
                 "($\\sigma_u=0.03$, $N=8{,}000$ paths)")
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.set_ylim(60, 102); ax.grid(alpha=0.18, lw=0.6)

    # panel (b): penalised utility by scenario
    ax2 = axes[1]
    w  = 0.25
    ax2.bar(scen_x-1.1*w, U_oracle, w, color=C["gz"],     alpha=0.8,
            label="Exp DDR oracle")
    ax2.bar(scen_x,        U_plugin, w, color=C["plugin"], alpha=0.8,
            label="Exp DDR plug-in")
    ax2.bar(scen_x+1.1*w,  U_bgz,    w, color=C["bgz"],    alpha=0.9,
            label="Bayes GZ (main)")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xticks(scen_x); ax2.set_xticklabels(scen_lb, fontsize=7)
    ax2.set_ylabel("Penalised utility $U(\\pi)$, $p=-2$")
    ax2.set_title("(b) Strategy comparison across uncertainty scenarios\n"
                  "($N=50{,}000$ paths, seed\\ 42)")
    ax2.legend(fontsize=6.5, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.18, lw=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig("../figures/fig5_monte_carlo.pdf")
    plt.close(); print("fig5 saved")


# ══════════════════════════════════════════════════════════════
# FIG 6  Robustness: kappa_rob, TC, rebalancing
# ══════════════════════════════════════════════════════════════
def make_fig6():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7))

    # panel (a): prior strength n_0 sensitivity (Bayes GZ at n_obs=5yr)
    n0_vals = [2, 5, 10, 20, 50, 100]
    U_bgz   = [-0.005,-0.007,-0.010,-0.014,-0.025,-0.041]
    U_plug  = [-0.003]*6   # plug-in is independent of n_0
    Sv_bgz  = [97.8, 97.4, 97.1, 96.6, 95.2, 93.3]

    ax = axes[0]
    axr = ax.twinx()
    ax.plot(n0_vals, U_bgz,  color=C["bgz"],    lw=2,   marker="o", ms=5,
            label="Bayes GZ")
    ax.axhline(U_plug[0], color=C["plugin"], ls="--", lw=1.4,
               label="Exp plug-in")
    ax.axhline(0, color="k", lw=0.5)
    axr.plot(n0_vals, Sv_bgz, color=C["green"], lw=1.4, ls=":",
             marker="s", ms=4, label="BGZ surv. \\%")
    ax.axvspan(10, 20, alpha=0.08, color=C["green"])
    ax.text(11, -0.038, "opt.\nrange", fontsize=6.5, color=C["green"])
    ax.set_xlabel("Prior strength $n_0$")
    ax.set_ylabel("$U(\\pi)$", color=C["bgz"])
    axr.set_ylabel("Survival \\%", color=C["green"])
    ax.set_title("(a) Sensitivity to prior $n_0$\n($n_\\mathrm{obs}=5$yr, $\\sigma_u=0.02$)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=6.5, framealpha=0.9, loc="lower left")
    ax.grid(alpha=0.18, lw=0.6)

    # panel (b): TC sensitivity (Bayes GZ vs plug-in)
    tc   = [0.0, 0.5, 1.0, 2.0, 5.0]
    Ua_t = [-0.010,-0.010,-0.010,-0.011,-0.013]
    Ue_t = [-0.003,-0.003,-0.004,-0.004,-0.004]

    ax2 = axes[1]
    ax2.plot(tc, Ua_t, color=C["bgz"],    lw=2,   marker="o", ms=5,
             label="Bayes GZ")
    ax2.plot(tc, Ue_t, color=C["plugin"], lw=1.8, marker="s", ms=4, ls="--",
             label="Exp plug-in")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("TC (bps per unit $\\Delta f$)")
    ax2.set_ylabel("$U(\\pi)$")
    ax2.set_title("(b) TC sensitivity\n($n_\\mathrm{obs}=5$yr, daily rebal.)")
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.grid(alpha=0.18, lw=0.6)

    # panel (c): rebalancing frequency (Bayes GZ at cold start 2yr)
    freqs  = ["Daily\n1d", "Weekly\n5d", "Biweek\n10d",
              "Monthly\n21d", "Qtly\n63d"]
    Ua_r   = [-0.084, -0.085, -0.092, -0.151, -0.250]
    Sv_r   = [ 93.8,   93.7,   93.0,   88.5,   80.2]

    ax3 = axes[2]
    clrs3 = [C["bgz"] if v > -0.10 else C["plugin"] for v in Ua_r]
    bars  = ax3.bar(range(5), Ua_r, color=clrs3, alpha=0.85)
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xticks(range(5)); ax3.set_xticklabels(freqs, fontsize=6.5)
    ax3.set_ylabel("$U(\\pi)$")
    ax3.set_title("(c) Rebalancing frequency\n(Bayes GZ, cold-start 2yr)")
    ax3.grid(axis="y", alpha=0.18, lw=0.6)

    fig.tight_layout(pad=0.7)
    fig.savefig("../figures/fig6_robustness.pdf")
    plt.close(); print("fig6 saved")


# ── run all ───────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("../figures", exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("\nDone – all figures saved to ../figures/")
