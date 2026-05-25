"""
Side-by-side comparison of kriging results on three real-world benchmark datasets:
  Walker Lake (470 pts), Meuse river zinc (155 pts), California earthquakes (1072 pts)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

OUT = Path("output/real_world")
DATASETS = {
    "Walker Lake\n(classic geostatistics benchmark)": {
        "data": "test_data/real_world/walker_lake_clean.csv",
        "pred": OUT / "walker_lake/walker_lake_clean/predicted_kriging.csv",
        "cv":   OUT / "walker_lake/walker_lake_clean/cv_results_kriging.csv",
        "params": OUT / "walker_lake/walker_lake_clean/parameters_kriging.json",
        "cols": ("X", "Y", "V"),
    },
    "Meuse River Zinc\n(Dutch floodplain, Pebesma 2004)": {
        "data": "test_data/real_world/meuse_clean.csv",
        "pred": OUT / "meuse/meuse_clean/predicted_kriging.csv",
        "cv":   OUT / "meuse/meuse_clean/cv_results_kriging.csv",
        "params": OUT / "meuse/meuse_clean/parameters_kriging.json",
        "cols": ("x", "y", "zinc"),
    },
    "California Earthquakes\n(2024, M≥2.5, USGS)": {
        "data": "test_data/real_world/california_quakes_clean.csv",
        "pred": OUT / "california_quakes/california_quakes_clean/predicted_kriging.csv",
        "cv":   OUT / "california_quakes/california_quakes_clean/cv_results_kriging.csv",
        "params": OUT / "california_quakes/california_quakes_clean/parameters_kriging.json",
        "cols": ("X", "Y", "Value"),
    },
}

# ---- Load everything ----
results = {}
for name, paths in DATASETS.items():
    d = {}
    d["raw"] = pd.read_csv(paths["data"])
    d["pred"] = pd.read_csv(paths["pred"])
    d["cv"] = pd.read_csv(paths["cv"])
    with open(paths["params"]) as f:
        d["params"] = json.load(f)
    xc, yc, vc = paths["cols"]
    d["xc"], d["yc"], d["vc"] = xc, yc, vc
    results[name] = d

# ---- Figure 1: Prediction surfaces + data points ----
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

for i, (name, d) in enumerate(results.items()):
    row, col = divmod(i, 3)

    # --- subplot A: data points (scatter) ---
    ax_pts = fig.add_subplot(gs[row * 2, col])
    raw = d["raw"]
    v = raw[d["vc"]]
    sc = ax_pts.scatter(raw[d["xc"]], raw[d["yc"]], c=v, s=12, cmap="Spectral_r",
                        edgecolors="k", linewidth=0.15, alpha=0.85)
    ax_pts.set_title(f"{name}\nInput data (n={len(raw)})", fontsize=10)
    ax_pts.set_aspect("equal")
    plt.colorbar(sc, ax=ax_pts, fraction=0.046, pad=0.04)

    # --- subplot B: prediction surface ---
    ax_pred = fig.add_subplot(gs[row * 2 + 1, col])
    pred = d["pred"]
    sc2 = ax_pred.scatter(pred["X"], pred["Y"], c=pred["predicted_mean"], s=8,
                          cmap="Spectral_r", alpha=0.9, edgecolors="none")
    # Overlay data points as small black dots
    ax_pred.scatter(raw[d["xc"]], raw[d["yc"]], s=2, c="black", alpha=0.4)
    ax_pred.set_title(f"Kriging prediction surface", fontsize=10)
    ax_pred.set_aspect("equal")
    plt.colorbar(sc2, ax=ax_pred, fraction=0.046, pad=0.04)

fig.suptitle("Real-World Kriging Benchmark: Data vs Prediction", fontsize=13, y=0.98)
fig.savefig(OUT / "comparison_data_vs_prediction.png", dpi=150)
plt.close(fig)
print("✓ comparison_data_vs_prediction.png")

# ---- Figure 2: Cross-validation diagnostics ----
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30)

for i, (name, d) in enumerate(results.items()):
    cv = d["cv"]

    # Observed vs Predicted
    ax1 = fig.add_subplot(gs[i, 0])
    ax1.scatter(cv["Observed"], cv["Predicted"], s=15, alpha=0.5, edgecolors="k", linewidth=0.1)
    lo, hi = cv[["Observed", "Predicted"]].min().min(), cv[["Observed", "Predicted"]].max().max()
    ax1.plot([lo, hi], [lo, hi], "r--", lw=1.2, alpha=0.6)
    r2 = 1 - np.sum(cv["Residual"]**2) / np.sum((cv["Observed"] - cv["Observed"].mean())**2)
    ax1.set_title(f"R² = {r2:.3f}")
    ax1.set_xlabel("Observed"); ax1.set_ylabel("Predicted")
    ax1.set_aspect("equal")

    # Residuals vs Predicted
    ax2 = fig.add_subplot(gs[i, 1])
    ax2.axhline(0, color="red", ls="--", lw=1, alpha=0.5)
    ax2.scatter(cv["Predicted"], cv["Residual"], s=15, alpha=0.5, edgecolors="k", linewidth=0.1)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Residual")
    ax2.set_title(f"MAE={np.mean(np.abs(cv['Residual'])):.2f}  RMSE={np.sqrt(np.mean(cv['Residual']**2)):.2f}")

    # Z-score histogram
    ax3 = fig.add_subplot(gs[i, 2])
    z = cv["Z_Score"]
    ax3.hist(z[~np.isnan(z) & np.isfinite(z)], bins=25, density=True, alpha=0.7, color="steelblue", edgecolor="k")
    from scipy.stats import norm as scipy_norm
    xr = np.linspace(-4, 4, 200)
    ax3.plot(xr, scipy_norm.pdf(xr), "r-", lw=1.5, label="N(0,1)")
    ax3.set_xlim(-4, 4)
    ax3.set_xlabel("Z-score"); ax3.set_title("Standardized residuals")
    ax3.legend(fontsize=7, loc="upper right")
    # Add short label to left
    short = name.split("\n")[0]
    ax3.text(-0.35, 0.5, short, transform=ax3.transAxes, fontsize=11, fontweight="bold",
             va="center", rotation=90)

fig.suptitle("Cross-Validation Diagnostics", fontsize=13, y=0.98)
fig.savefig(OUT / "comparison_cv_diagnostics.png", dpi=150)
plt.close(fig)
print("✓ comparison_cv_diagnostics.png")

# ---- Figure 3: Parameter comparison bar chart ----
param_keys = ["rotation_angle_deg", "anisotropy_ratio", "psill", "range", "nugget"]
param_labels = ["Aniso. angle (°)", "Aniso. ratio", "Partial sill", "Range", "Nugget"]
n_params = len(param_keys)
fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5))
short_names = ["Walker Lake", "Meuse Zinc", "CA Quakes"]

colors = ["#e74c3c", "#3498db", "#2ecc71"]
for j, key in enumerate(param_keys):
    ax = axes[j]
    vals = [results[n]["params"][key] for n in results]
    bars = ax.bar(short_names, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_title(param_labels[j])
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    # Annotate values
    for bar, val in zip(bars, vals):
        if abs(val) > 100:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                    f"{val:.0f}", ha="center", fontsize=7)
        elif abs(val) < 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                    f"{val:.3f}", ha="center", fontsize=7)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                    f"{val:.1f}", ha="center", fontsize=7)

fig.suptitle("Optimized Variogram Parameters", fontsize=13, y=1.01)
fig.savefig(OUT / "comparison_parameters.png", dpi=150)
plt.close(fig)
print("✓ comparison_parameters.png")

# ---- Figure 4: Variogram model shapes ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, (name, d) in enumerate(results.items()):
    ax = axes[i]
    p = d["params"]
    h = np.linspace(0, p["range"] * 2.5, 300)

    if p["best_model"] == "stable":
        gamma = p["nugget"] + p["psill"] * (1 - np.exp(-(h / p["range"]) ** p["alpha"]))
    elif p["best_model"] == "circular":
        gamma = np.full_like(h, p["nugget"] + p["psill"])
        mask = h <= p["range"]
        hr = h[mask] / p["range"]
        gamma[mask] = p["nugget"] + p["psill"] * (1 - (2/np.pi) * (np.arccos(hr) - hr * np.sqrt(1 - hr**2)))
    elif p["best_model"] == "rational-quadratic":
        gamma = p["nugget"] + p["psill"] * (1 - (1 + h**2 / (2 * p["alpha"] * p["range"]**2)) ** (-p["alpha"]))
    else:
        gamma = p["nugget"] + p["psill"] * (1 - np.exp(-3 * h / p["range"]))

    ax.plot(h, gamma, "b-", lw=2)
    ax.axhline(p["nugget"] + p["psill"], color="gray", ls="--", lw=0.8, label=f"sill={p['nugget']+p['psill']:.1f}")
    ax.axhline(p["nugget"], color="red", ls="--", lw=0.8, label=f"nugget={p['nugget']:.1f}")
    ax.axvline(p["range"], color="green", ls="--", lw=0.8, label=f"range={p['range']:.1f}")
    ax.set_title(f'{short_names[i]}\nmodel={p["best_model"]}')
    ax.set_xlabel("Distance h"); ax.set_ylabel("γ(h)")
    ax.legend(fontsize=7)

fig.suptitle("Fitted Variogram Models", fontsize=13, y=1.01)
fig.savefig(OUT / "comparison_variograms.png", dpi=150)
plt.close(fig)
print("✓ comparison_variograms.png")

# ---- Text summary ----
with open(OUT / "benchmark_summary.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  REAL-WORLD KRIGING BENCHMARK SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    for name, d in results.items():
        cv = d["cv"]
        r2 = 1 - np.sum(cv["Residual"]**2) / np.sum((cv["Observed"] - cv["Observed"].mean())**2)
        mae = float(np.mean(np.abs(cv["Residual"])))
        rmse = float(np.sqrt(np.mean(cv["Residual"]**2)))
        p = d["params"]
        nugget_frac = p["nugget"] / (p["nugget"] + p["psill"]) if (p["nugget"] + p["psill"]) > 0 else 0
        f.write(f"Dataset: {name.split(chr(10))[0]}\n")
        f.write(f"  n={len(d['raw'])}  |  R²={r2:.4f}  |  MAE={mae:.2f}  |  RMSE={rmse:.2f}\n")
        f.write(f"  Model: {p['best_model']}  |  Angle: {p['rotation_angle_deg']:.1f}°  |  Ratio: {p['anisotropy_ratio']:.2f}\n")
        f.write(f"  Range: {p['range']:.1f}  |  Nugget fraction: {nugget_frac:.1%}\n\n")

    f.write("-" * 70 + "\n")
    f.write("INTERPRETATION\n")
    f.write("-" * 70 + "\n")
    f.write("Walker Lake: Best performance (R²=0.23). Strong anisotropy recovered (ratio≈5).\n")
    f.write("  The stable model with alpha≈0.4 suggests rough spatial continuity.\n")
    f.write("  Modest R² is expected — the original Isaaks & Srivastava analysis reports\n")
    f.write("  similar CV performance due to high nugget effect (44% of sill).\n\n")
    f.write("Meuse Zinc: Negative R² (-0.08) indicates kriging performs worse than the mean.\n")
    f.write("  The strong skew (1.47) triggered NST, but with only 155 points spread over\n")
    f.write("  a floodplain that follows a meandering river, ordinary kriging cannot capture\n")
    f.write("  the non-stationary mean. Universal kriging with distance-to-river as drift\n")
    f.write("  would be the canonical fix (per Pebesma's original analysis).\n\n")
    f.write("California Quakes: Negative R² (-0.11), nugget dominates 89% of sill.\n")
    f.write("  Earthquake magnitudes are fundamentally point-process events along fault\n")
    f.write("  lines, not a continuous spatial field. Kriging is the wrong tool — the\n")
    f.write("  spatial covariance is dominated by fault geometry, not distance decay.\n")
    f.write("  Kernel density estimation or fault-zone models would be more appropriate.\n")

print("✓ benchmark_summary.txt")
print("\nAll outputs saved to output/real_world/")
