"""
fig7_stress_tests.py — stress test figure for paper
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "lines.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})

C = {"oracle": "#2c3e50", "plugin": "#c0392b", "bgz": "#0f3460",
     "green": "#27ae60", "orange": "#e67e22", "gray": "#7f8c8d"}

# ── Data (N=50,000) ──────────────────────────────────────────────────────
env_labels = [
    "A. GBM\n(baseline)",
    "B. Markov\nregime",
    "C. Student-t\n(ν=4)",
    "D. Jump\ndiffusion",
    "E. GARCH\nvol cluster.",
]

U_oracle = [+0.0858, +0.0337, +0.0517, -0.7399, +0.1167]
U_plugin = [-0.1713, -0.1113, -0.4592, -0.8256, -0.5636]
U_bgz    = [-0.0864, -0.0546, -0.4205, -0.8669, -0.6454]

S_oracle = [100.0, 100.0,  98.2,  62.5, 100.0]
S_plugin = [ 89.5,  93.5,  75.2,  58.6,  71.0]
S_bgz    = [ 93.7,  96.3,  77.1,  56.5,  67.3]

delta_U    = [b-p for b,p in zip(U_bgz, U_plugin)]
delta_surv = [b-p for b,p in zip(S_bgz, S_plugin)]

fig = plt.figure(figsize=(7.2, 8.0))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

x = np.arange(len(env_labels))
w = 0.26

# ── Panel (a): Penalised utility ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(x - w, U_oracle, w, color=C["oracle"], alpha=0.75, label="Exp DDR oracle")
ax1.bar(x,     U_plugin, w, color=C["plugin"], alpha=0.85, label="Exp DDR plug-in")
ax1.bar(x + w, U_bgz,    w, color=C["bgz"],    alpha=0.85, label="Bayes GZ [main]")
ax1.axhline(0, color="k", lw=0.7)
ax1.set_xticks(x); ax1.set_xticklabels(env_labels, fontsize=8)
ax1.set_ylabel("Penalised utility $U(\\pi)$, $p=-2$")
ax1.set_title("(a) Strategy comparison across return environments\n"
              "($n_{\\rm obs}=2$\\,yr, $\\sigma_u=0.03$, $N=50{,}000$ paths, ruin penalty $p=-2$)",
              fontsize=9)
ax1.legend(framealpha=0.9, loc="lower right")
ax1.grid(axis="y", alpha=0.2, lw=0.6)

# Colour bands for regime context
for i, (label, col, alpha) in enumerate([
    ("GBM valid", C["green"],  0.05),
    ("BGZ holds", C["green"],  0.05),
    ("BGZ holds", C["green"],  0.05),
    ("BGZ fails", C["plugin"], 0.08),
    ("BGZ fails", C["plugin"], 0.08),
]):
    ax1.axvspan(i-0.5, i+0.5, alpha=alpha, color=col, zorder=0)

# ── Panel (b): Survival rates ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(x, S_oracle, "k--o", lw=1.5, ms=5, label="Exp DDR oracle")
ax2.plot(x, S_plugin, color=C["plugin"], marker="s", lw=2, ms=5,
         label="Exp DDR plug-in")
ax2.plot(x, S_bgz,    color=C["bgz"],    marker="^", lw=2, ms=5,
         label="Bayes GZ [main]")
ax2.fill_between(x, S_plugin, S_bgz,
                 where=[b>p for b,p in zip(S_bgz,S_plugin)],
                 alpha=0.15, color=C["green"], label="BGZ advantage")
ax2.fill_between(x, S_bgz, S_plugin,
                 where=[b<p for b,p in zip(S_bgz,S_plugin)],
                 alpha=0.15, color=C["plugin"], label="BGZ disadvantage")
ax2.axhline(95, color=C["orange"], ls=":", lw=1)
ax2.set_xticks(x); ax2.set_xticklabels(env_labels, fontsize=8)
ax2.set_ylabel("Survival rate (\\%)")
ax2.set_title("(b) Survival at 20\\% drawdown barrier across environments")
ax2.legend(framealpha=0.9, ncol=3, fontsize=7)
ax2.grid(alpha=0.2, lw=0.6); ax2.set_ylim(45, 105)

# ── Panel (c): ΔU = BGZ − plug-in ────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
colors_du = [C["bgz"] if d > 0 else C["plugin"] for d in delta_U]
bars = ax3.bar(x, delta_U, color=colors_du, alpha=0.85, width=0.6)
ax3.axhline(0, color="k", lw=0.7)
for bar, d in zip(bars, delta_U):
    ax3.text(bar.get_x()+bar.get_width()/2,
             d + 0.003*np.sign(d),
             f"{d:+.3f}", ha="center",
             va="bottom" if d > 0 else "top",
             fontsize=7)
ax3.set_xticks(x); ax3.set_xticklabels(env_labels, fontsize=7)
ax3.set_ylabel("$\\Delta U = U_{\\rm BGZ} - U_{\\rm plug-in}$")
ax3.set_title("(c) Bayes GZ advantage\n(positive = BGZ wins)")
ax3.grid(axis="y", alpha=0.2, lw=0.6)
# Annotate regions
ax3.text(0.5,  0.065, "✓ BGZ\nadvantage", ha="center", fontsize=7, color=C["bgz"])
ax3.text(3.5, -0.065, "✗ BGZ\ndisadvantage", ha="center", fontsize=7, color=C["plugin"])

# ── Panel (d): ΔSurv ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
colors_ds = [C["bgz"] if d > 0 else C["plugin"] for d in delta_surv]
bars2 = ax4.bar(x, delta_surv, color=colors_ds, alpha=0.85, width=0.6)
ax4.axhline(0, color="k", lw=0.7)
for bar, d in zip(bars2, delta_surv):
    ax4.text(bar.get_x()+bar.get_width()/2,
             d + 0.1*np.sign(d),
             f"{d:+.1f}pp", ha="center",
             va="bottom" if d > 0 else "top",
             fontsize=7)
ax4.set_xticks(x); ax4.set_xticklabels(env_labels, fontsize=7)
ax4.set_ylabel("$\\Delta$Surv (pp)")
ax4.set_title("(d) Survival advantage\n(positive = BGZ wins)")
ax4.grid(axis="y", alpha=0.2, lw=0.6)

fig.suptitle(
    "Stress tests: Bayes GZ vs. Exp DDR plug-in across return environments\n"
    "BGZ dominates under regime switching and fat tails; "
    "plug-in is better under extreme jumps and GARCH",
    fontsize=9, fontweight="bold", y=1.01
)

plt.savefig("/home/claude/paper3/figures/fig7_stress_tests.pdf")
plt.close()
print("fig7 saved")
