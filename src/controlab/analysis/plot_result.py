"""Render the robustness-breadth and trade-off figures from eval_results.parquet."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from controlab.eval.stats_processing import (
    probability_of_improvement,
    robustness_curve,
    score_matrix,
)

font = {"size": 8}
plt.style.use("default")
plt.rc("text", usetex=True)
plt.rc("font", **font, family="serif")

# analysis/ is src/controlab/analysis -> parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette; fixed order per bucket
COLORS = {"nominal": "blue", "interpolation": "orange", "extrapolation": "green"}
LABELS = {"nominal": "Nominal", "interpolation": "Interpolation", "extrapolation": "Extrapolation"}
BUCKETS = ["nominal", "interpolation", "extrapolation"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_parquet(REPO_ROOT / "output" / "eval_results.parquet")
curves = {b: robustness_curve(df, b, reps=5000) for b in BUCKETS}

# ---- Figure 1: robustness-breadth (IQM return vs breadth, 3 buckets, 95% CI) ----
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
for bucket in BUCKETS:
    c = curves[bucket]
    ax.fill_between(c.alpha, c.lo, c.hi, color=COLORS[bucket], alpha=0.18, linewidth=0)
    ax.plot(c.alpha, c.iqm, "-o", color=COLORS[bucket], lw=2, ms=5, label=LABELS[bucket])
    ax.annotate(
        LABELS[bucket],
        (c.alpha.iloc[-1], c.iqm.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=COLORS[bucket],
        fontsize=10,
        fontweight="bold",
    )

ax.set_xlabel(r"Randomization breadth $\alpha$")
ax.set_ylabel(r"IQM episodic return  ($95\%$ CI)")
ax.set_title("Zero-shot transfer vs. domain-randomization breadth")
ax.set_ylim(0, 1050)
ax.grid(True, alpha=0.3, lw=0.6)
ax.legend(frameon=False, loc="center left")
fig.tight_layout()
fig.savefig(OUT / "robustness_breadth.png", bbox_inches="tight")

# ---- Figure 2: probability of improvement over the no-DR baseline (extrapolation) ----
BASELINE = 0.0
POI_BUCKET = "extrapolation"
compare_alphas = [a for a in sorted(df["alpha"].unique())]

y = score_matrix(df, BASELINE, POI_BUCKET)
poi_rows = []
for alpha in compare_alphas:
    x = score_matrix(df, alpha, POI_BUCKET)
    prob, lo, hi = probability_of_improvement(x, y, reps=5000)
    poi_rows.append((alpha, prob, lo, hi))

fig3, ax3 = plt.subplots(figsize=(6.6, 4.2), dpi=150)
xpos = list(range(len(poi_rows)))
ps = [r[1] for r in poi_rows]
yerr = [[r[1] - r[2] for r in poi_rows], [r[3] - r[1] for r in poi_rows]]
ax3.errorbar(xpos, ps, yerr=yerr, fmt="o", color="black", ms=7, capsize=4, lw=2)
ax3.axhline(0.5, color="0.5", ls="--", lw=1)  # 0.5 = no difference
ax3.set_xticks(xpos)
ax3.set_xticklabels([rf"{r[0]:g}" for r in poi_rows])
ax3.set_ylim(0.3, 1.02)
ax3.set_ylabel(rf"P(improvement vs. $\alpha=0$  |  {POI_BUCKET} )")
ax3.set_title("Probability of improvement over the no-DR baseline")
ax3.grid(True, axis="y", alpha=0.3, lw=0.6)
fig3.tight_layout()
fig3.savefig(OUT / "probability_of_improvement.png", bbox_inches="tight")

print("plotted:", OUT / "robustness_breadth.png")
print("plotted:", OUT / "probability_of_improvement.png")
