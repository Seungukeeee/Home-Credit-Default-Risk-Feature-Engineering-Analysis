"""
Threshold Sweep Visualization
==============================
Plots FN, FP, and Net Benefit across thresholds to show why
threshold optimization matters more than feature engineering.

Run this locally after the combined_experiment.py / threshold_analysis.py
results are saved.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────
# 0. Load data
# ──────────────────────────────────────────────
# Use threshold_analysis.csv (baseline-only sweep) for the cleanest story
df = pd.read_csv("./results/threshold_analysis.csv")

# ──────────────────────────────────────────────
# 1. Style
# ──────────────────────────────────────────────
plt.style.use("default")
fig, ax1 = plt.subplots(figsize=(10, 6))

COLOR_FN   = "#E63946"  # red
COLOR_FP   = "#457B9D"  # blue
COLOR_NET  = "#2A9D8F"  # teal

# ──────────────────────────────────────────────
# 2. Left axis — FN & FP counts
# ──────────────────────────────────────────────
ax1.plot(df["threshold"], df["FN"], marker="o", color=COLOR_FN,
         linewidth=2, label="False Negatives (missed defaults)")
ax1.plot(df["threshold"], df["FP"], marker="s", color=COLOR_FP,
         linewidth=2, label="False Positives (rejected good payers)")
ax1.set_xlabel("Decision Threshold", fontsize=12)
ax1.set_ylabel("Case Count", fontsize=12)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.invert_xaxis()  # so it reads left(0.5) -> right(0.05), matching the narrative direction
ax1.grid(alpha=0.3)

# ──────────────────────────────────────────────
# 3. Right axis — Net Benefit
# ──────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.plot(df["threshold"], df["savings_usd"] if "savings_usd" in df.columns else df.get("net_benefit_usd"),
          marker="^", color=COLOR_NET, linewidth=2.5, linestyle="--",
          label="Net Benefit (USD)")
ax2.set_ylabel("Net Benefit (USD)", fontsize=12, color=COLOR_NET)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:,.0f}M"))
ax2.axhline(0, color="gray", linewidth=1, linestyle=":")
ax2.tick_params(axis="y", labelcolor=COLOR_NET)

# mark the peak
peak_idx = df["savings_usd"].idxmax() if "savings_usd" in df.columns else df["net_benefit_usd"].idxmax()
peak_th  = df.loc[peak_idx, "threshold"]
peak_val = df.loc[peak_idx, "savings_usd"] if "savings_usd" in df.columns else df.loc[peak_idx, "net_benefit_usd"]
ax2.annotate(f"Peak: ${peak_val/1e6:,.0f}M\n@ threshold={peak_th}",
             xy=(peak_th, peak_val), xytext=(peak_th - 0.05, peak_val * 0.85),
             fontsize=10, fontweight="bold", color=COLOR_NET,
             arrowprops=dict(arrowstyle="->", color=COLOR_NET))

# ──────────────────────────────────────────────
# 4. Title & legend
# ──────────────────────────────────────────────
plt.title("Threshold Optimization: FN/FP Tradeoff and Net Business Benefit",
          fontsize=14, fontweight="bold", pad=15)

# combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: threshold_sweep.png")
