"""GroundTruthWindow — QDialog showing 2×2 validation comparison plots."""

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
    QPushButton, QMessageBox, QFileDialog,
)
from ui_pyside.mpl_canvas import MplCanvas


class GroundTruthWindow(QDialog):
    """Model validation against external ground truth data.

    Shows a 2×2 grid: Predicted vs Observed scatter, Error Map,
    Metrics table, and Residual Histogram.
    """

    def __init__(self, metrics: dict, residuals: np.ndarray,
                 gt_obs: np.ndarray, gt_pred: np.ndarray,
                 gt_X: np.ndarray, gt_Y: np.ndarray,
                 gt_col: str = "Value", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ground Truth Validation")
        self.setMinimumSize(900, 650)
        self.resize(1000, 700)

        self._metrics = metrics
        self._residuals = residuals
        self._gt_obs = gt_obs
        self._gt_pred = gt_pred
        self._gt_X = gt_X
        self._gt_Y = gt_Y
        self._gt_col = gt_col

        layout = QVBoxLayout(self)

        # 2×2 plot grid
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self._canvases = {}
        positions = [
            (0, 0, "scatter", "Predicted vs Observed"),
            (0, 1, "error_map", "Spatial Error Map"),
            (1, 0, "metrics", "Metrics"),
            (1, 1, "histogram", "Residual Distribution"),
        ]
        for row, col, key, title in positions:
            canvas = MplCanvas(figsize=(4.5, 3.2), dpi=100)
            self._canvases[key] = canvas
            grid.addWidget(canvas, row, col)

        layout.addLayout(grid)

        # Bottom buttons
        btn_row = QHBoxLayout()
        export_btn = QPushButton("Export…")
        export_btn.setToolTip("Save comparison figure and metrics CSV")
        export_btn.clicked.connect(self._export)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(export_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._draw()

    # ------------------------------------------------------------------
    def _draw(self):
        self._draw_scatter()
        self._draw_error_map()
        self._draw_metrics_panel()
        self._draw_histogram()

    def _draw_scatter(self):
        canvas = self._canvases["scatter"]
        fig = canvas.fig; fig.clear()
        ax = fig.add_subplot(111)
        ax.scatter(self._gt_obs, self._gt_pred, alpha=0.6, s=36,
                   color="#1f77b4", edgecolors="white", linewidth=0.5, zorder=3)
        # 1:1 reference line
        lims = [min(self._gt_obs.min(), self._gt_pred.min()),
                max(self._gt_obs.max(), self._gt_pred.max())]
        pad = (lims[1] - lims[0]) * 0.05
        ax.plot([lims[0] - pad, lims[1] + pad],
                [lims[0] - pad, lims[1] + pad],
                "r--", lw=1.5, alpha=0.7, label="1:1 line")
        # R² annotation
        r2 = self._metrics.get("r2", float("nan"))
        ax.text(0.03, 0.97, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=11, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"Observed ({self._gt_col})", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title("Predicted vs Observed", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(labelsize=8)
        ax.grid(True, ls=":", alpha=0.4)
        canvas.draw_idle()

    def _draw_error_map(self):
        canvas = self._canvases["error_map"]
        fig = canvas.fig; fig.clear()
        ax = fig.add_subplot(111)
        res = self._residuals
        abs_res = np.abs(res)
        max_abs = max(abs_res.max(), 1e-9)
        sizes = 20 + 120 * (abs_res / max_abs)
        colors = np.where(res >= 0, "#d62728", "#1f77b4")  # red=over, blue=under
        sc = ax.scatter(self._gt_X, self._gt_Y, c=colors, s=sizes,
                        alpha=0.75, edgecolors="white", linewidth=0.4)
        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d62728", alpha=0.75, label="Over-prediction (+)"),
            Patch(facecolor="#1f77b4", alpha=0.75, label="Under-prediction (−)"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
        ax.set_xlabel("X", fontsize=9); ax.set_ylabel("Y", fontsize=9)
        ax.set_title("Spatial Error Map (size ∝ |error|)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        canvas.draw_idle()

    def _draw_metrics_panel(self):
        canvas = self._canvases["metrics"]
        fig = canvas.fig; fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")
        m = self._metrics
        rows = [
            ("MAE",       f"{m.get('mae', float('nan')):.4f}"),
            ("RMSE",      f"{m.get('rmse', float('nan')):.4f}"),
            ("R²",        f"{m.get('r2', float('nan')):.4f}"),
            ("Bias",      f"{m.get('bias', float('nan')):.4f}"),
            ("mean SSPE", f"{m.get('mean_sspe', float('nan')):.4f}"),
            ("RMSS",      f"{m.get('rmss', float('nan')):.4f}"),
            ("N",         f"{m.get('n', 0)}"),
        ]
        y = 0.92
        ax.text(0.05, y + 0.04, "Validation Metrics", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        for label, value in rows:
            ax.text(0.08, y, label, transform=ax.transAxes,
                    fontsize=11, va="top", fontweight="bold", color="#555")
            ax.text(0.50, y, value, transform=ax.transAxes,
                    fontsize=11, va="top", family="monospace")
            y -= 0.11
        canvas.draw_idle()

    def _draw_histogram(self):
        canvas = self._canvases["histogram"]
        fig = canvas.fig; fig.clear()
        ax = fig.add_subplot(111)
        res = self._residuals
        ax.hist(res, bins=max(12, int(len(res) ** 0.5)), density=True,
                color="steelblue", alpha=0.6, edgecolor="white", linewidth=0.5)
        # Normal PDF overlay
        from scipy.stats import norm as scipy_norm
        mu, sigma = float(np.mean(res)), float(np.std(res))
        if sigma > 0:
            x = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 200)
            ax.plot(x, scipy_norm.pdf(x, mu, sigma), "r-", lw=2,
                    label=f"N({mu:.2f}, {sigma:.2f})")
        ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Residual (Predicted − Observed)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title("Residual Distribution", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, ls=":", alpha=0.4, axis="y")
        canvas.draw_idle()

    # ------------------------------------------------------------------
    def _export(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Export Ground Truth Results")
        if not folder:
            return
        saved = []
        try:
            # Save the full figure as PNG
            fig_png = Path(folder) / "ground_truth_comparison.png"
            # Create a combined figure for export
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            big_fig = self._canvases["scatter"].fig
            big_fig.savefig(fig_png, dpi=150, bbox_inches="tight")
            saved.append(fig_png.name)

            # Save metrics CSV
            import csv
            metrics_csv = Path(folder) / "ground_truth_metrics.csv"
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._metrics.keys())
                writer.writeheader()
                writer.writerow(self._metrics)
            saved.append(metrics_csv.name)

            QMessageBox.information(self, "Export complete",
                "Saved:\n" + "\n".join(saved))
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
