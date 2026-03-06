"""
compare_systems.py
Comprehensive comparison of SPD-RAG, Normal RAG, and Agentic RAG evaluation results.
Usage: uv run python benchmark/loong/compare_systems.py
"""

import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "SPD-RAG":      ROOT / "spd_rag_25pro_flash_p3_other_metrics_cleanup.jsonl",
    "Normal RAG":   ROOT / "normal_rag_25pro_p3_other_metrics_cleanup.jsonl",
    "Agentic RAG":  ROOT / "agentic_rag_25pro_p3_other_metrics_cleanup.jsonl",
}

PALETTE = {
    "SPD-RAG":     "#2563EB",
    "Normal RAG":  "#DC2626",
    "Agentic RAG": "#16A34A",
}

OUT_DIR = Path(__file__).parent / "comparison_plots"
OUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.80
BAR_WIDTH = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)

dfs = {}
for name, path in FILES.items():
    if not path.exists():
        print(f"[WARN] File not found, skipping: {path}")
        continue
    dfs[name] = load(path)

systems = list(dfs.keys())
colors  = [PALETTE[s] for s in systems]

# Shared task ids (tasks that appear in all systems)
all_ids = set.intersection(*(set(df["id"]) for df in dfs.values()))
print(f"Tasks per system: { {s: len(df) for s, df in dfs.items()} }")
print(f"Tasks shared across all systems: {len(all_ids)}")

# Restrict to shared tasks for fair comparisons
shared_dfs = {s: df[df["id"].isin(all_ids)].copy() for s, df in dfs.items()}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def bar_group(ax, categories, values_by_system, ylabel="", title="", fmt=".1f"):
    x = np.arange(len(categories))
    n = len(systems)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * BAR_WIDTH
    all_vals = [v for sys_name in systems
                for v in values_by_system[sys_name].values() if v == v]  # exclude NaN
    y_max = max(all_vals) if all_vals else 1
    label_offset = y_max * 0.03  # 3% of the data range
    for sys_name, offset in zip(systems, offsets):
        vals = [values_by_system[sys_name].get(cat, 0) for cat in categories]
        bars = ax.bar(x + offset, vals, BAR_WIDTH, label=sys_name,
                      color=PALETTE[sys_name], alpha=ALPHA)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, y_max * 1.15)  # ensure labels fit inside axes
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary_rows = []
for sys_name, df in shared_dfs.items():
    row = {
        "System":          sys_name,
        "N":               len(df),
        "Mean Score":      df["score"].mean(),
        "Median Score":    df["score"].median(),
        "Perfect (=100)":  (df["score"] == 100).sum(),
        "Perfect %":       (df["score"] == 100).mean() * 100,
        "Zero %":          (df["score"] == 0).mean() * 100,
        "Mean Tokens":     df["ls_total_tokens"].mean(),
        "Mean Cost $":     df["ls_cost_usd"].mean(),
        "Mean Latency s":  df["ls_latency"].mean(),
        "Total Cost $":    df["ls_cost_usd"].sum(),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("System")
pd.set_option("display.float_format", "{:.3f}".format)
print(summary_df.T.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. OVERALL SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Plot 1] Score distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Score Distributions by System", fontweight="bold", fontsize=13)

for ax, (sys_name, df) in zip(axes, shared_dfs.items()):
    scores = df["score"]
    ax.hist(scores, bins=20, color=PALETTE[sys_name], alpha=0.8, edgecolor="white")
    ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean {scores.mean():.1f}")
    ax.axvline(scores.median(), color="gray", linestyle=":", linewidth=1.5,
               label=f"Median {scores.median():.1f}")
    ax.set_title(sys_name, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
save(fig, "01_score_distributions")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BOX PLOT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 2] Score box plots...")
fig, ax = plt.subplots(figsize=(7, 5))
data  = [shared_dfs[s]["score"].values for s in systems]
bp = ax.boxplot(data, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(ALPHA)
ax.set_xticklabels(systems)
ax.set_ylabel("Score (0–100)")
ax.set_title("Score Box Plot Comparison", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save(fig, "02_score_boxplot")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SCORE BY LEVEL_NAME
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 3] Score by level_name...")
levels = sorted(set().union(*(df["level_name"].unique() for df in shared_dfs.values())))
by_level = {s: {lv: df[df["level_name"] == lv]["score"].mean()
                for lv in levels} for s, df in shared_dfs.items()}

fig, ax = plt.subplots(figsize=(max(8, len(levels) * 1.8), 5))
bar_group(ax, levels, by_level, ylabel="Mean Score", title="Mean Score by Level")
plt.tight_layout()
save(fig, "03_score_by_level")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SCORE BY TYPE
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 4] Score by type...")
types = sorted(set().union(*(df["type"].unique() for df in shared_dfs.values())))
by_type = {s: {t: df[df["type"] == t]["score"].mean()
               for t in types} for s, df in shared_dfs.items()}

fig, ax = plt.subplots(figsize=(max(7, len(types) * 1.8), 5))
bar_group(ax, types, by_type, ylabel="Mean Score", title="Mean Score by Document Type")
plt.tight_layout()
save(fig, "04_score_by_type")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SCORE BY LEVEL × TYPE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 5] Score heatmaps per system...")
fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), max(4, len(levels) * 0.7 + 1)))
fig.suptitle("Mean Score — Level × Type Heatmap", fontweight="bold", fontsize=13)

for ax, sys_name in zip(axes, systems):
    df = shared_dfs[sys_name]
    pivot = df.pivot_table(index="level_name", columns="type", values="score", aggfunc="mean")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(sys_name, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                        color="black" if 20 < val < 80 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
save(fig, "05_heatmap_level_type")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PERFECT SCORE RATE BY LEVEL
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 6] Perfect score rate by level...")
perfect_by_level = {s: {lv: (df[df["level_name"] == lv]["score"] == 100).mean() * 100
                          for lv in levels} for s, df in shared_dfs.items()}

fig, ax = plt.subplots(figsize=(max(8, len(levels) * 1.8), 5))
bar_group(ax, levels, perfect_by_level, ylabel="Perfect Score %",
          title="Perfect Score Rate (score=100) by Level", fmt=".0f")
plt.tight_layout()
save(fig, "06_perfect_rate_by_level")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SCORE vs. NUM_DOCS SCATTER
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 7] Score vs. num_docs scatter...")
fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 4), sharey=True)
fig.suptitle("Score vs. Number of Documents", fontweight="bold", fontsize=13)
for ax, sys_name in zip(axes, systems):
    df = shared_dfs[sys_name]
    ax.scatter(df["num_docs"], df["score"], color=PALETTE[sys_name], alpha=0.5, s=25)
    m, b, r, p, _ = stats.linregress(df["num_docs"], df["score"])
    xs = np.linspace(df["num_docs"].min(), df["num_docs"].max(), 100)
    ax.plot(xs, m * xs + b, color="black", linewidth=1.5,
            label=f"r={r:.2f}, p={p:.3f}")
    ax.set_title(sys_name, fontweight="bold")
    ax.set_xlabel("Num Docs")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, "07_score_vs_numdocs")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EFFICIENCY METRICS
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 8] Efficiency — score per $ and score per 1k tokens...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Score per dollar — aggregate ratio (total score / total cost), not mean of per-task ratios
ax = axes[0]
sppd = {s: df["score"].sum() / df["ls_cost_usd"].sum() for s, df in shared_dfs.items()}
bars = ax.bar(systems, [sppd[s] for s in systems], color=colors, alpha=ALPHA)
for bar, v in zip(bars, [sppd[s] for s in systems]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score per Dollar")
ax.set_title("Score Efficiency (total score / total cost $)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Score per 1k tokens — aggregate ratio
ax = axes[1]
sppt = {s: df["score"].sum() / (df["ls_total_tokens"].sum() / 1000) for s, df in shared_dfs.items()}
bars = ax.bar(systems, [sppt[s] for s in systems], color=colors, alpha=ALPHA)
for bar, v in zip(bars, [sppt[s] for s in systems]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score per 1k Tokens")
ax.set_title("Token Efficiency (total score / total 1k tokens)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save(fig, "08_efficiency")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TOKEN USAGE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 9] Token usage breakdown...")
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Token Usage per System", fontweight="bold", fontsize=13)

metrics = [("ls_input_tokens", "Mean Input Tokens"),
           ("ls_output_tokens", "Mean Output Tokens"),
           ("ls_total_tokens", "Mean Total Tokens")]

for ax, (col, label) in zip(axes, metrics):
    vals = [shared_dfs[s][col].mean() for s in systems]
    bars = ax.bar(systems, vals, color=colors, alpha=ALPHA)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(label, fontweight="bold")
    ax.set_ylabel("Tokens")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save(fig, "09_token_breakdown")

# ─────────────────────────────────────────────────────────────────────────────
# 11. COST & LATENCY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 10] Cost and latency box plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cost & Latency Distributions", fontweight="bold", fontsize=13)

for ax, (col, label) in zip(axes, [("ls_cost_usd", "Cost per Task (USD)"),
                                     ("ls_latency",   "Latency per Task (s)")]):
    data = [shared_dfs[s][col].values for s in systems]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(ALPHA)
    ax.set_xticklabels(systems)
    ax.set_ylabel(label)
    ax.set_title(label, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save(fig, "10_cost_latency_boxplots")

# ─────────────────────────────────────────────────────────────────────────────
# 12. MEAN COST & LATENCY BY LEVEL
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 11] Cost and latency by level...")
fig, axes = plt.subplots(1, 2, figsize=(max(12, len(levels) * 2.5), 5))

cost_by_level = {s: {lv: df[df["level_name"] == lv]["ls_cost_usd"].mean()
                     for lv in levels} for s, df in shared_dfs.items()}
lat_by_level  = {s: {lv: df[df["level_name"] == lv]["ls_latency"].mean()
                     for lv in levels} for s, df in shared_dfs.items()}

bar_group(axes[0], levels, cost_by_level, ylabel="Mean Cost (USD)",
          title="Mean Cost by Level", fmt=".3f")
bar_group(axes[1], levels, lat_by_level,  ylabel="Mean Latency (s)",
          title="Mean Latency by Level", fmt=".1f")

plt.tight_layout()
save(fig, "11_cost_latency_by_level")

# ─────────────────────────────────────────────────────────────────────────────
# 13. PER-TASK SCORE COMPARISON (PAIRED)
# ─────────────────────────────────────────────────────────────────────────────

print("[Plot 12] Per-task score scatter matrix...")
shared_ids = sorted(all_ids)
paired = {s: shared_dfs[s].set_index("id")["score"].reindex(shared_ids)
          for s in systems}

pairs = [(systems[i], systems[j]) for i in range(len(systems)) for j in range(i + 1, len(systems))]
fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
if len(pairs) == 1:
    axes = [axes]
fig.suptitle("Per-Task Score Scatter (Paired)", fontweight="bold", fontsize=13)

for ax, (sa, sb) in zip(axes, pairs):
    xa, xb = paired[sa].values, paired[sb].values
    ax.scatter(xa, xb, alpha=0.4, s=20, color="#6366F1")
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="y = x")
    m, b, r, p, _ = stats.linregress(xa[~np.isnan(xa + xb)], xb[~np.isnan(xa + xb)])
    ax.set_xlabel(f"{sa} Score")
    ax.set_ylabel(f"{sb} Score")
    ax.set_title(f"{sa} vs {sb}\nr={r:.2f}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # Shade quadrants
    ax.axvline(50, color="gray", linewidth=0.5, alpha=0.4)
    ax.axhline(50, color="gray", linewidth=0.5, alpha=0.4)

plt.tight_layout()
save(fig, "12_per_task_scatter")

# ─────────────────────────────────────────────────────────────────────────────
# 14. WIN RATE TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("WIN RATE (on shared tasks)")
print("=" * 70)

score_mat = pd.DataFrame({s: shared_dfs[s].set_index("id")["score"].reindex(sorted(all_ids))
                           for s in systems})

for sa in systems:
    for sb in systems:
        if sa == sb:
            continue
        wins   = (score_mat[sa] > score_mat[sb]).sum()
        ties   = (score_mat[sa] == score_mat[sb]).sum()
        losses = (score_mat[sa] < score_mat[sb]).sum()
        print(f"  {sa} vs {sb}: W={wins}  T={ties}  L={losses}  "
              f"WR={wins / len(score_mat) * 100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 15. STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("WILCOXON SIGNED-RANK TEST (score, paired by task)")
print("=" * 70)

for sa, sb in pairs:
    xa = score_mat[sa].dropna()
    xb = score_mat[sb].reindex(xa.index).dropna()
    common = xa.index.intersection(xb.index)
    stat, p = stats.wilcoxon(xa[common], xb[common])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {sa} vs {sb}: W={stat:.1f}, p={p:.4f} {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. DASHBOARD SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Plot 13] Dashboard summary...")
fig = plt.figure(figsize=(18, 10))
fig.suptitle("System Comparison Dashboard", fontweight="bold", fontsize=15)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# Mean score bar
ax0 = fig.add_subplot(gs[0, 0])
vals = [shared_dfs[s]["score"].mean() for s in systems]
bars = ax0.bar(systems, vals, color=colors, alpha=ALPHA)
for bar, v in zip(bars, vals):
    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax0.set_title("Mean Score", fontweight="bold")
ax0.set_ylim(0, 100)
ax0.grid(axis="y", alpha=0.3)

# Perfect %
ax1 = fig.add_subplot(gs[0, 1])
vals = [(shared_dfs[s]["score"] == 100).mean() * 100 for s in systems]
bars = ax1.bar(systems, vals, color=colors, alpha=ALPHA)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
ax1.set_title("Perfect Score %", fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# Mean cost
ax2 = fig.add_subplot(gs[0, 2])
vals = [shared_dfs[s]["ls_cost_usd"].mean() for s in systems]
bars = ax2.bar(systems, vals, color=colors, alpha=ALPHA)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f"${v:.3f}", ha="center", va="bottom", fontsize=9)
ax2.set_title("Mean Cost / Task", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

# Mean latency
ax3 = fig.add_subplot(gs[0, 3])
vals = [shared_dfs[s]["ls_latency"].mean() for s in systems]
bars = ax3.bar(systems, vals, color=colors, alpha=ALPHA)
for bar, v in zip(bars, vals):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
ax3.set_title("Mean Latency / Task", fontweight="bold")
ax3.grid(axis="y", alpha=0.3)

# Score by level
ax4 = fig.add_subplot(gs[1, :2])
bar_group(ax4, levels, by_level, ylabel="Mean Score", title="Mean Score by Level")

# Score by type
ax5 = fig.add_subplot(gs[1, 2:])
bar_group(ax5, types, by_type, ylabel="Mean Score", title="Mean Score by Type")

save(fig, "00_dashboard")

print(f"\nAll plots saved to: {OUT_DIR}")
print("Done.")
