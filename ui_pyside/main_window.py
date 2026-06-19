"""GeospatialApp — main QMainWindow for the PySide6 interpolation engine UI."""
import sys, os, csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ui_pyside  # noqa: F401 — DLL safety first

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QStatusBar, QLabel,
    QMenuBar, QMenu, QMessageBox, QWidget, QGridLayout, QVBoxLayout,
    QRadioButton, QCheckBox, QComboBox, QPushButton, QButtonGroup,
    QFileDialog, QDialog, QDialogButtonBox,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence

from ui_pyside.accordion_sidebar import AccordionSidebar
from ui_pyside.animated_slider import AnimatedSlider
from ui_pyside.dockable_plot import DockablePlot
from ui_pyside.file_picker import FilePicker
from ui_pyside.log_console import LogConsole
from ui_pyside.workspace_controller import WorkspaceController
from ui_pyside.theme import apply_theme
from ui.variogram_panel import VARIOGRAM_MODELS


class GeospatialApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interpolation Engine")
        self.resize(1400, 900)

        self._ctrl = WorkspaceController(self)

        splitter = QSplitter(Qt.Horizontal)
        self._sidebar = self._build_sidebar()
        splitter.addWidget(self._sidebar)
        self._plot_grid = self._build_plot_grid()
        splitter.addWidget(self._plot_grid)
        splitter.setSizes([320, 1080])
        self.setCentralWidget(splitter)

        self._build_menus()
        self._build_statusbar()
        self._wire_signals()
        self._restore_state()

    # ── sidebar ──────────────────────────────────────────────────────
    def _build_sidebar(self) -> AccordionSidebar:
        sb = AccordionSidebar()

        # Section 1: Data & Setup
        sec_data = sb.addSection("Data & Setup")
        dw = QWidget(); dl = QVBoxLayout(dw); dl.setContentsMargins(4, 4, 4, 4)
        self._file_picker = FilePicker()
        dl.addWidget(self._file_picker)
        self._col_x = QComboBox(); self._col_y = QComboBox(); self._col_z = QComboBox()
        for lbl, cb in [("X column:", self._col_x), ("Y column:", self._col_y),
                         ("Value column:", self._col_z)]:
            row = QWidget(); rl = QVBoxLayout(row); rl.setContentsMargins(0, 2, 0, 2)
            rl.addWidget(QLabel(lbl)); rl.addWidget(cb); dl.addWidget(row)
        eng_w = QWidget(); el = QVBoxLayout(eng_w); el.setContentsMargins(0, 4, 0, 0)
        el.addWidget(QLabel("Engine:"))
        self._eng_group = QButtonGroup(self)
        self._radio_krig = QRadioButton("Ordinary Kriging")
        self._radio_gp = QRadioButton("Gaussian Process")
        self._radio_krig.setChecked(True)
        self._eng_group.addButton(self._radio_krig, 0)
        self._eng_group.addButton(self._radio_gp, 1)
        el.addWidget(self._radio_krig); el.addWidget(self._radio_gp)
        dl.addWidget(eng_w)
        sec_data.setContent(dw)
        sec_data.expand()

        # Section 2: Variogram Controls
        sec_vario = sb.addSection("Variogram Controls")
        vw = QWidget(); vl = QVBoxLayout(vw); vl.setContentsMargins(4, 4, 4, 4)
        self._model_dropdown = QComboBox()
        self._model_dropdown.addItems(list(VARIOGRAM_MODELS.keys()))
        vl.addWidget(QLabel("Model:")); vl.addWidget(self._model_dropdown)
        self._sliders = {}
        for name, mn, mx, df in [
            ("Range", 1, 5000, 300), ("Sill (psill)", 0.001, 50, 5.0),
            ("Nugget", 0, 20, 0.5), ("Angle (°)", 0, 180, 0),
            ("Anisotropy ×", 1, 15, 1), ("Alpha", 0.1, 2.0, 1.0),
        ]:
            sl = AnimatedSlider(label=name, min_val=mn, max_val=mx, default=df)
            self._sliders[name] = sl
            vl.addWidget(sl)
        self._live_cb = QCheckBox("Live update")
        self._live_cb.setChecked(True)
        vl.addWidget(self._live_cb)
        btn_row = QWidget(); bl = QVBoxLayout(btn_row); bl.setContentsMargins(0, 4, 0, 0)
        self._run_btn = QPushButton("▶  Run full-res")
        self._auto_btn = QPushButton("⟳  Auto-fit")
        self._export_btn = QPushButton("\U0001F4BE  Export…")
        for b in (self._run_btn, self._auto_btn, self._export_btn):
            bl.addWidget(b)
        vl.addWidget(btn_row)
        sec_vario.setContent(vw)

        # Section 3: Engine Options (collapsed)
        sec_eng = sb.addSection("Engine Options")
        ew = QWidget(); el2 = QVBoxLayout(ew)
        el2.addWidget(QLabel("GP optimization trials:"))
        self._gp_trials_cb = QComboBox()
        self._gp_trials_cb.addItems(["100", "300", "500", "1000"])
        self._gp_trials_cb.setCurrentText("300")
        el2.addWidget(self._gp_trials_cb)
        sec_eng.setContent(ew)

        # Section 4: Log (collapsed)
        sec_log = sb.addSection("Log")
        self._log = LogConsole()
        sec_log.setContent(self._log)

        return sb

    # ── plot grid ────────────────────────────────────────────────────
    def _build_plot_grid(self) -> QWidget:
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(4)
        self._plots = {}
        for (row, col, name, objname) in [
            (0, 0, "Prediction Surface", "SurfaceDock"),
            (0, 1, "Uncertainty (std)", "UncertaintyDock"),
            (1, 0, "Variogram Fit", "VarioDock"),
            (1, 1, "CV Dashboard", "CVDock"),
        ]:
            dp = DockablePlot(title=name, object_name=objname)
            self._plots[name] = dp
            grid.addWidget(dp, row, col)
        return container

    # ── menus ────────────────────────────────────────────────────────
    def _build_menus(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        file_menu.addAction("&Open Input File…",
                            lambda: self._file_picker._browse(),
                            QKeySequence.Open)
        file_menu.addAction("Set &Output Folder…", self._browse_output)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, QKeySequence("Ctrl+Q"))
        view_menu = mb.addMenu("&View")
        for name in self._plots:
            view_menu.addAction(self._plots[name].toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction("&Reset Layout", self._reset_layout)
        analysis_menu = mb.addMenu("&Analysis")
        analysis_menu.addAction("&Run Full-res", self._ctrl.run_full,
                                QKeySequence("Ctrl+R"))
        analysis_menu.addAction("Auto-&fit", self._ctrl.auto_fit,
                                QKeySequence("Ctrl+Shift+R"))
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("&About", self._about)

    def _build_statusbar(self):
        self._metrics_label = QLabel(
            "MAE —    RMSE —    R² —    mean_SSPE —    RMSS —")
        self._status_label = QLabel("Ready")
        self.statusBar().addWidget(self._metrics_label, 1)
        self.statusBar().addPermanentWidget(self._status_label)

    # ── wiring ───────────────────────────────────────────────────────
    def _wire_signals(self):
        c = self._ctrl
        self._file_picker.fileSelected.connect(self._on_file_selected)
        for cb in (self._col_x, self._col_y, self._col_z):
            cb.currentTextChanged.connect(lambda: self._try_load())
        self._eng_group.buttonToggled.connect(self._on_engine_changed)
        self._model_dropdown.currentTextChanged.connect(self._fire_sliders)
        for sl in self._sliders.values():
            sl.valueChanged.connect(lambda v: self._fire_sliders())
        self._live_cb.toggled.connect(c.set_live)
        self._run_btn.clicked.connect(c.run_full)
        self._auto_btn.clicked.connect(c.auto_fit)
        self._export_btn.clicked.connect(self._export)
        c.logLine.connect(self._log.appendLine)
        c.statusMessage.connect(self._status_label.setText)
        c.metricsUpdated.connect(self._on_metrics)
        c.resultReady.connect(self._on_result)

    def _on_file_selected(self, path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            headers = csv.DictReader(f).fieldnames or []
        for cb in (self._col_x, self._col_y, self._col_z):
            cb.clear(); cb.addItems(headers)
        if len(headers) >= 3:
            self._col_x.setCurrentText(headers[0])
            self._col_y.setCurrentText(headers[1])
            self._col_z.setCurrentText(headers[-1])
        self._try_load()

    def _try_load(self):
        fp = self._file_picker.path()
        cx = self._col_x.currentText()
        cy = self._col_y.currentText()
        cz = self._col_z.currentText()
        if fp and cx and cy and cz:
            self._ctrl.load_data(fp, cx, cy, cz)
            self._sidebar.expandSection(1)

    def _on_engine_changed(self, btn, checked):
        if not checked:
            return
        mode = "gp" if btn is self._radio_gp else "kriging"
        self._ctrl.set_engine(mode)
        is_krig = mode == "kriging"
        self._model_dropdown.setVisible(is_krig)
        for name in ("Sill (psill)", "Nugget", "Alpha"):
            if name in self._sliders:
                self._sliders[name].setVisible(is_krig)

    def _fire_sliders(self, *_):
        preset = {"model": self._model_dropdown.currentText()}
        for n, sl in self._sliders.items():
            if n == "Range": preset["range"] = sl.value()
            elif n == "Sill (psill)": preset["psill"] = sl.value()
            elif n == "Nugget": preset["nugget"] = sl.value()
            elif n == "Angle (°)": preset["angle_deg"] = sl.value()
            elif n == "Anisotropy ×": preset["anisotropy_ratio"] = sl.value()
            elif n == "Alpha": preset["alpha"] = sl.value()
        if self._ctrl._engine == "gp":
            preset = {"kernel_type": "matern_52",
                       "length_scale_major": self._sliders["Range"].value(),
                       "anisotropy_ratio": self._sliders["Anisotropy ×"].value(),
                       "angle_deg": self._sliders["Angle (°)"].value()}
        self._ctrl.on_slider_change(preset)
        self._redraw_vario_fit(preset)

    def _on_result(self, result):
        grid = result.get("grid")
        if grid is not None and "mean" in grid:
            self._draw_surface(grid)
        cv_df = result.get("cv_df")
        if cv_df is not None:
            self._draw_cv(cv_df)

    def _draw_surface(self, grid):
        for dp_name, data_key, cmap, title in [
            ("Prediction Surface", "mean", "viridis", "Predicted Mean"),
            ("Uncertainty (std)", "std", "magma_r", "Uncertainty (std)"),
        ]:
            dp = self._plots.get(dp_name)
            if dp is None or "xv" not in grid:
                continue
            fig = dp.canvas.fig; fig.clear(); ax = fig.add_subplot(111)
            try:
                c = ax.contourf(grid["xv"], grid["yv"], grid[data_key],
                                 levels=30, cmap=cmap, extend="both")
                fig.colorbar(c, ax=ax, fraction=0.046)
            except Exception:
                ax.text(0.5, 0.5, "no surface", ha="center", va="center",
                        transform=ax.transAxes)
            hull = grid.get("hull")
            if hull is not None and len(hull) > 0:
                ax.plot(hull[:, 0], hull[:, 1], "w-", lw=1.0, alpha=0.7)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            dp.canvas.draw_idle()

    def _draw_cv(self, cv_df):
        from utils import plot_cv_dashboard
        dp = self._plots.get("CV Dashboard")
        if dp is None:
            return
        try:
            plot_cv_dashboard(cv_df, engine_name=self._ctrl._engine.upper(),
                              scenario_name="", fig=dp.canvas.fig)
        except Exception as exc:
            fig = dp.canvas.fig; fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"CV plot failed:\n{exc}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
        dp.canvas.draw_idle()

    def _redraw_vario_fit(self, preset):
        import numpy as np
        from ui.variogram_panel import compute_empirical_variogram, compute_model_curve
        dp = self._plots.get("Variogram Fit")
        if dp is None or self._ctrl._X is None:
            return
        fig = dp.canvas.fig; fig.clear(); ax = fig.add_subplot(111)
        ax.set_xlabel("Lag distance"); ax.set_ylabel("Semivariance")
        ax.set_title("Variogram Fit")
        lags, sv = compute_empirical_variogram(self._ctrl._X, self._ctrl._y)
        if len(lags):
            ax.scatter(lags, sv, color="#1f77b4", s=28, zorder=3, label="Empirical")
            h = np.linspace(0, lags[-1] * 1.2, 200)
            gam = compute_model_curve(
                preset.get("model", "spherical"), h,
                preset.get("range", 300), preset.get("psill", 5.0),
                preset.get("nugget", 0.5), preset.get("alpha", 1.0))
            ax.plot(h, gam, color="#d62728", lw=2, label=preset.get("model", "?"))
            ax.legend(fontsize=9)
        dp.canvas.draw_idle()

    def _on_metrics(self, m: dict):
        def g(k):
            v = m.get(k)
            return f"{v:.4g}" if isinstance(v, (int, float)) and v == v else "—"
        self._metrics_label.setText(
            f"MAE {g('mae')}    RMSE {g('rmse')}    "
            f"R² {g('r2')}    mean_SSPE {g('mean_sspe')}    RMSS {g('rmss')}")

    def _export(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Export results")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Export to folder:"))
        fig_cb = QCheckBox("Figures (PNG)"); fig_cb.setChecked(True)
        grid_cb = QCheckBox("Predicted grid (.npz)"); grid_cb.setChecked(True)
        cv_cb = QCheckBox("CV results (.csv)"); cv_cb.setChecked(True)
        layout.addWidget(fig_cb); layout.addWidget(grid_cb); layout.addWidget(cv_cb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.Accepted:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not folder:
            return
        saved = []
        if fig_cb.isChecked():
            for name in self._plots:
                p = Path(folder) / f"{name.replace(' ', '_').lower()}.png"
                self._plots[name].canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
                saved.append(p.name)
        ctrl_saved = self._ctrl.export(folder, grid_cb.isChecked(), cv_cb.isChecked())
        if ctrl_saved:
            saved.extend(ctrl_saved)
        QMessageBox.information(self, "Export complete",
            "Saved:\n" + "\n".join(saved) if saved else "Nothing selected.")

    def _about(self):
        QMessageBox.about(self, "About",
            "Interpolation Engine\nGeostatistics · Kriging · Gaussian Process")

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Set output folder")
        if d:
            self._ctrl._state["output_dir"] = d

    def _reset_layout(self):
        for dp in self._plots.values():
            dp.setFloating(False)
        self._sidebar.expandSection(0)

    def _restore_state(self):
        s = QSettings("InterpEngine", "GeospatialApp")
        geo = s.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        ws = s.value("windowState")
        if ws:
            self.restoreState(ws)

    def closeEvent(self, event):
        s = QSettings("InterpEngine", "GeospatialApp")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = GeospatialApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
