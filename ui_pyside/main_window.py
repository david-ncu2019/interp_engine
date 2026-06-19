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
    QFileDialog, QDialog, QDialogButtonBox, QSpinBox, QTabWidget,
    QProgressBar,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence

from ui_pyside.accordion_sidebar import AccordionSidebar
from ui_pyside.animated_slider import AnimatedSlider
from ui_pyside.dockable_plot import PlotPanel
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
        self.setMinimumSize(800, 500)

        self._ctrl = WorkspaceController(self)
        self._vario_tabs = None        # QTabWidget for variogram tabs
        self._vario_canvases = {}      # {tab_name: MplCanvas}
        self._dir_cache = None         # cached directional variogram results

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
        self._radio_krig.setToolTip(
            "Classical geostatistical interpolation. Fast, fits a variogram\n"
            "model to your data, produces a predicted surface + kriging\n"
            "variance. Good default for most spatial datasets.")
        self._radio_gp = QRadioButton("Gaussian Process")
        self._radio_gp.setToolTip(
            "Probabilistic machine learning approach. Fits a covariance\n"
            "kernel by maximizing the marginal likelihood. Slower but\n"
            "produces a full predictive distribution with per-point\n"
            "uncertainty. Better for small to medium datasets.")
        self._radio_krig.setChecked(True)
        self._eng_group.addButton(self._radio_krig, 0)
        self._eng_group.addButton(self._radio_gp, 1)
        el.addWidget(self._radio_krig); el.addWidget(self._radio_gp)
        dl.addWidget(eng_w)
        sec_data.setContent(dw)
        sec_data.expand()

        # Section 2: Preprocessing
        sec_preproc = sb.addSection("Preprocessing")
        pw = QWidget(); pl = QVBoxLayout(pw); pl.setContentsMargins(4, 4, 4, 4)

        # Detrend controls
        detrend_w = QWidget(); dl2 = QVBoxLayout(detrend_w); dl2.setContentsMargins(0, 0, 0, 0)
        dl2.addWidget(QLabel("Detrend:"))
        self._detrend_cb = QCheckBox("Enabled")
        self._detrend_cb.setToolTip(
            "Subtract a polynomial trend surface before fitting.\n"
            "Useful when your data has a regional gradient\n"
            "(e.g., groundwater flow, elevation slope).")
        self._detrend_cb.setChecked(False)
        dl2.addWidget(self._detrend_cb)

        order_row = QWidget(); orl = QVBoxLayout(order_row); orl.setContentsMargins(20, 0, 0, 0)
        orl.addWidget(QLabel("Polynomial degree:"))
        self._detrend_order_cb = QComboBox()
        self._detrend_order_cb.addItems(["1 — linear plane", "2 — quadratic", "3 — cubic"])
        self._detrend_order_cb.setToolTip(
            "Order of the polynomial trend surface.\n"
            "1 = flat dipping plane (most common).\n"
            "2 = bowl/dome shape. 3 = complex undulations.")
        orl.addWidget(self._detrend_order_cb)
        dl2.addWidget(order_row)

        self._detrend_auto_cb = QCheckBox("Auto-detect (F-test)")
        self._detrend_auto_cb.setToolTip(
            "When checked: the engine decides whether to detrend\n"
            "based on a statistical F-test of the polynomial fit.\n"
            "The 'Enabled' checkbox above is ignored when auto-detect is on.")
        self._detrend_auto_cb.setChecked(False)
        dl2.addWidget(self._detrend_auto_cb)
        pl.addWidget(detrend_w)

        # NST controls
        nst_row = QWidget(); nrl = QVBoxLayout(nst_row); nrl.setContentsMargins(0, 6, 0, 0)
        nrl.addWidget(QLabel("Normal Score Transform (NST):"))
        self._nst_combo = QComboBox()
        self._nst_combo.addItems(["Off", "On", "Auto"])
        self._nst_combo.setToolTip(
            "Transform data to a standard normal distribution before fitting.\n"
            "Essential for strongly skewed data (e.g., ore grades, rainfall).\n"
            "Off = never transform. On = always transform.\n"
            "Auto = let the engine decide (Shapiro-Wilk normality test).")
        nrl.addWidget(self._nst_combo)
        pl.addWidget(nst_row)

        # Info notice
        info = QLabel(
            "These settings only apply to \"Run Interpolation\".\n"
            "Live preview and \"Optimize Parameters\" always\n"
            "use raw data to keep the variogram plot consistent.")
        info.setWordWrap(True)
        info.setStyleSheet(
            "QLabel { color: palette(placeholder-text); font-style: italic; "
            "padding: 4px; font-size: 10px; }")
        pl.addWidget(info)

        sec_preproc.setContent(pw)

        # Section 3: Variogram Controls
        sec_vario = sb.addSection("Variogram Controls")
        vw = QWidget(); vl = QVBoxLayout(vw); vl.setContentsMargins(4, 4, 4, 4)
        # Kriging: variogram model dropdown
        self._krig_model_label = QLabel("Variogram Model:")
        self._krig_model_combo = QComboBox()
        self._krig_model_combo.setToolTip(
            "The mathematical model for the variogram curve.\n"
            "Different models have different shapes near the origin\n"
            "(linear, S-shaped, asymptotic) — try several to compare.")
        for m in VARIOGRAM_MODELS:
            label = (f"{m} (variogram)"
                     if m.startswith("matern") else m)
            self._krig_model_combo.addItem(label, m)
        vl.addWidget(self._krig_model_label)
        vl.addWidget(self._krig_model_combo)
        # GP: kernel type dropdown (hidden by default)
        self._gp_kernel_label = QLabel("Kernel Type:")
        self._gp_kernel_combo = QComboBox()
        self._gp_kernel_combo.addItems([
            "matern_32 — Matérn-3/2  (rough, C¹)",
            "matern_52 — Matérn-5/2  (moderate, C²)",
            "rbf — RBF  (smooth, C∞)",
        ])
        self._gp_kernel_combo.setToolTip(
            "The covariance kernel determines how smooth the predicted\n"
            "surface is. Matérn-3/2 = rougher (realistic for many natural\n"
            "phenomena). RBF = infinitely smooth (less common in nature).")
        self._gp_kernel_label.setVisible(False)
        self._gp_kernel_combo.setVisible(False)
        vl.addWidget(self._gp_kernel_label)
        vl.addWidget(self._gp_kernel_combo)
        # GP notice (visible only for GP)
        self._gp_notice = QLabel(
            "GP uses marginal likelihood, not variogram fitting.\n"
            "The empirical variogram is shown for reference only.")
        self._gp_notice.setWordWrap(True)
        self._gp_notice.setStyleSheet(
            "QLabel { color: palette(placeholder-text); font-style: italic; "
            "padding: 4px; font-size: 10px; }")
        self._gp_notice.setVisible(False)
        vl.addWidget(self._gp_notice)
        # Number of lags (Kriging only)
        self._nlags_label = QLabel("Number of lags:")
        self._nlags_spin = QSpinBox()
        self._nlags_spin.setRange(4, 50)
        self._nlags_spin.setValue(12)
        self._nlags_spin.setToolTip(
            "How many distance bins to use when computing the empirical\n"
            "variogram from your data. More lags = finer detail but\n"
            "noisier. Fewer lags = smoother but may miss short-range patterns.")
        self._nlags_spin.valueChanged.connect(self._fire_sliders)
        vl.addWidget(self._nlags_label)
        vl.addWidget(self._nlags_spin)
        self._sliders = {}
        _slider_tooltips = {
            "Range": "The distance at which the variogram levels off.\n"
                     "Points farther apart than this range are no longer\n"
                     "spatially correlated — they're independent.",
            "Sill (psill)": "The total structured variance — how much the data\n"
                            "varies due to spatial position alone (excluding\n"
                            "pure noise from the nugget).",
            "Nugget": "Variance at zero distance — measurement error +\n"
                      "variation at scales smaller than your sample spacing.\n"
                      "A high nugget means noisy data or fine-scale structure.",
            "Angle (°)": "The direction of maximum spatial continuity.\n"
                         "0° = East-West, 90° = North-South.\n"
                         "Only matters when anisotropy is > 1.",
            "Anisotropy ×": "How much longer the range is in the major direction\n"
                            "vs. the minor direction. 1.0 = isotropic (same in\n"
                            "all directions). > 1 = anisotropic.",
            "Alpha": "Extra shape parameter for stable and rational-quadratic\n"
                     "models. Controls the curve's steepness near the origin.\n"
                     "α < 1 = sharper, α > 1 = smoother.",
        }
        for name, mn, mx, df in [
            ("Range", 1, 5000, 300), ("Sill (psill)", 0.001, 50, 5.0),
            ("Nugget", 0, 20, 0.5), ("Angle (°)", 0, 180, 0),
            ("Anisotropy ×", 1, 15, 1), ("Alpha", 0.1, 2.0, 1.0),
        ]:
            sl = AnimatedSlider(label=name, min_val=mn, max_val=mx, default=df)
            self._sliders[name] = sl
            if name in _slider_tooltips:
                sl.setToolTip(_slider_tooltips[name])
            vl.addWidget(sl)
        self._live_cb = QCheckBox("Live update")
        self._live_cb.setChecked(True)
        self._live_cb.setToolTip(
            "When ON: dragging any slider instantly re-renders the prediction\n"
            "surface and uncertainty map at low resolution (live preview).\n"
            "When OFF: click 'Run Interpolation' to see results.")
        vl.addWidget(self._live_cb)
        btn_row = QWidget(); bl = QVBoxLayout(btn_row); bl.setContentsMargins(0, 4, 0, 0)
        self._run_btn = QPushButton("▶  Run Interpolation")
        self._run_btn.setToolTip(
            "Run the full kriging / GP pipeline with your current slider values.\n"
            "This computes the prediction surface, cross-validation metrics\n"
            "(MAE, RMSE, R², mean_SSPE, RMSS), and the CV dashboard.")
        self._auto_btn = QPushButton("⟳  Optimize Parameters")
        self._auto_btn.setToolTip(
            "Automatically find the best variogram / kernel parameters by\n"
            "optimizing against your data (no sliders needed). After it finishes,\n"
            "your sliders will update to the optimized values.")
        self._export_btn = QPushButton("\U0001F4BE  Export…")
        self._export_btn.setToolTip(
            "Save results to a folder. Nothing is written to disk until you\n"
            "click this. Choose what to export: figures, grid, CV results.")
        for b in (self._run_btn, self._auto_btn, self._export_btn):
            bl.addWidget(b)
        vl.addWidget(btn_row)
        sec_vario.setContent(vw)

        # Section 4: Engine Options (collapsed)
        sec_eng = sb.addSection("Engine Options")
        ew = QWidget(); el2 = QVBoxLayout(ew)
        el2.addWidget(QLabel("GP optimization trials:"))
        self._gp_trials_cb = QComboBox()
        self._gp_trials_cb.addItems(["100", "300", "500", "1000"])
        self._gp_trials_cb.setCurrentText("300")
        el2.addWidget(self._gp_trials_cb)
        sec_eng.setContent(ew)

        # Section 5: Log (collapsed)
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
        for (row, col, name) in [
            (0, 0, "Prediction Surface"),
            (0, 1, "Uncertainty (std)"),
            (1, 1, "CV Dashboard"),
        ]:
            pp = PlotPanel(title=name)
            self._plots[name] = pp
            grid.addWidget(pp, row, col)
        # (1, 0) — variogram tabs (3-tab widget)
        self._vario_tabs = self._build_vario_tabs()
        grid.addWidget(self._vario_tabs, 1, 0)
        return container

    def _build_vario_tabs(self) -> QTabWidget:
        from ui_pyside.mpl_canvas import MplCanvas
        tabs = QTabWidget()
        for tab_name in ["Omnidirectional", "Directional 15°", "Anisotropy Rose"]:
            canvas = MplCanvas(figsize=(5.0, 3.5), dpi=96)
            self._vario_canvases[tab_name] = canvas
            tabs.addTab(canvas, tab_name)
        return tabs

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
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)        # hidden when idle
        self.statusBar().addWidget(self._metrics_label, 1)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().addPermanentWidget(self._status_label)

    # ── wiring ───────────────────────────────────────────────────────
    def _wire_signals(self):
        c = self._ctrl
        self._file_picker.fileSelected.connect(self._on_file_selected)
        for cb in (self._col_x, self._col_y, self._col_z):
            cb.currentTextChanged.connect(lambda: self._try_load())
        self._eng_group.buttonToggled.connect(self._on_engine_changed)
        self._krig_model_combo.currentTextChanged.connect(self._fire_sliders)
        self._gp_kernel_combo.currentTextChanged.connect(self._fire_sliders)
        for sl in self._sliders.values():
            sl.valueChanged.connect(lambda v: self._fire_sliders())
        self._live_cb.toggled.connect(c.set_live)
        self._run_btn.clicked.connect(c.run_full)
        self._auto_btn.clicked.connect(c.auto_fit)
        self._export_btn.clicked.connect(self._export)
        self._detrend_cb.toggled.connect(self._on_preproc_changed)
        self._detrend_order_cb.currentIndexChanged.connect(self._on_preproc_changed)
        self._detrend_auto_cb.toggled.connect(self._on_preproc_changed)
        self._nst_combo.currentIndexChanged.connect(self._on_preproc_changed)
        c.logLine.connect(self._log.appendLine)
        c.statusMessage.connect(self._status_label.setText)
        c.dataLoaded.connect(self._on_data_loaded)
        c.metricsUpdated.connect(self._on_metrics)
        c.paramsReady.connect(self._on_params_ready)
        c.resultReady.connect(self._on_result)
        c.progressChanged.connect(self._on_progress)
        c.busyStarted.connect(self._on_busy_started)
        c.busyFinished.connect(self._on_busy_finished)

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
            self._sidebar.expandSection(2)  # Variogram Controls (now section index 2)

    def _on_engine_changed(self, btn, checked):
        if not checked:
            return
        mode = "gp" if btn is self._radio_gp else "kriging"
        self._ctrl.set_engine(mode)
        is_krig = mode == "kriging"
        self._krig_model_label.setVisible(is_krig)
        self._krig_model_combo.setVisible(is_krig)
        self._nlags_label.setVisible(is_krig)
        self._nlags_spin.setVisible(is_krig)
        self._gp_kernel_label.setVisible(not is_krig)
        self._gp_kernel_combo.setVisible(not is_krig)
        self._gp_notice.setVisible(not is_krig)
        for name in ("Sill (psill)", "Nugget", "Alpha"):
            if name in self._sliders:
                self._sliders[name].setVisible(is_krig)

    def _on_preproc_changed(self, *_):
        detrend_enabled = self._detrend_cb.isChecked()
        detrend_order = self._detrend_order_cb.currentIndex() + 1  # 1, 2, 3
        detrend_auto = self._detrend_auto_cb.isChecked()
        nst_mode = self._nst_combo.currentIndex()  # 0=Off, 1=On, 2=Auto
        nst_enabled = {0: False, 1: True, 2: None}[nst_mode]
        self._ctrl.set_preprocessing(detrend_enabled, detrend_order,
                                     detrend_auto, nst_enabled)

    def _current_preset(self) -> dict:
        """Build the preset dict from the current engine + slider/combo state."""
        if self._ctrl._engine == "gp":
            # Parse kernel type from display name like "matern_52 — Matérn-5/2  (moderate, C²)"
            kt_text = self._gp_kernel_combo.currentText()
            kt = kt_text.split(" ")[0] if kt_text else "matern_52"
            return {"kernel_type": kt,
                    "length_scale_major": self._sliders["Range"].value(),
                    "anisotropy_ratio": self._sliders["Anisotropy ×"].value(),
                    "angle_deg": self._sliders["Angle (°)"].value()}
        preset = {"model": self._krig_model_combo.currentData()}
        for n, sl in self._sliders.items():
            if n == "Range": preset["range"] = sl.value()
            elif n == "Sill (psill)": preset["psill"] = sl.value()
            elif n == "Nugget": preset["nugget"] = sl.value()
            elif n == "Angle (°)": preset["angle_deg"] = sl.value()
            elif n == "Anisotropy ×": preset["anisotropy_ratio"] = sl.value()
            elif n == "Alpha": preset["alpha"] = sl.value()
        return preset

    def _draw_variogram_tabs(self, preset=None):
        """Redraw all three variogram tabs from the current data + preset.
        Does NOT start a prediction — safe to call on bare data load."""
        if preset is None:
            preset = self._current_preset()
        self._compute_directional()          # recompute dir_cache (n_lags may have changed)
        self._redraw_omnidirectional(preset)
        self._redraw_directional()
        self._redraw_anisotropy_rose(preset)

    def _fire_sliders(self, *_):
        preset = self._current_preset()
        self._ctrl._state["kriging_n_lags"] = self._nlags_spin.value()
        self._ctrl.on_slider_change(preset)   # debounced live preview (genuine interaction)
        self._draw_variogram_tabs(preset)

    def _on_data_loaded(self):
        """On bare data load: show the empirical variogram FIRST (not a prediction).
        The user hasn't fit or run anything yet — draw the variogram tabs and put a
        neutral placeholder in the prediction/uncertainty panels."""
        self._draw_variogram_tabs()
        self._show_prediction_placeholders()

    def _show_prediction_placeholders(self):
        msg = "Adjust a parameter or click Run / Optimize to predict"
        for dp_name in ("Prediction Surface", "Uncertainty (std)"):
            dp = self._plots.get(dp_name)
            if dp is None:
                continue
            fig = dp.canvas.fig; fig.clear(); ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray",
                    wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            dp.canvas.draw_idle()

    def _on_result(self, result):
        grid = result.get("grid")
        if grid is not None and "mean" in grid:
            self._draw_surface(grid)
        cv_df = result.get("cv_df")
        if cv_df is not None:
            self._draw_cv(cv_df)

    def _draw_surface(self, grid):
        # Accept both full-run keys (xv/yv) and live-preview keys (X_grid/Y_grid)
        xg = grid.get("xv", grid.get("X_grid"))
        yg = grid.get("yv", grid.get("Y_grid"))
        if xg is None:
            return
        for dp_name, data_key, cmap, title in [
            ("Prediction Surface", "mean", "viridis", "Predicted Mean"),
            ("Uncertainty (std)", "std", "magma_r", "Uncertainty (std)"),
        ]:
            dp = self._plots.get(dp_name)
            if dp is None:
                continue
            fig = dp.canvas.fig; fig.clear(); ax = fig.add_subplot(111)
            try:
                c = ax.contourf(xg, yg, grid[data_key],
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

    # ── variogram tabs ─────────────────────────────────────────────────
    def _compute_directional(self):
        """Compute 12 directional empirical variograms (15° intervals) and cache."""
        if self._ctrl._X is None:
            self._dir_cache = None
            return
        from utils import compute_empirical_variogram as utils_variogram
        directions = list(range(0, 180, 15))  # 0, 15, …, 165
        n_lags = self._nlags_spin.value()
        try:
            self._dir_cache = utils_variogram(
                self._ctrl._X, self._ctrl._y,
                n_lags=n_lags, directions=directions)
        except Exception:
            self._dir_cache = None

    def _redraw_omnidirectional(self, preset):
        import numpy as np
        # Use the SAME empirical variogram the optimizer fits (utils, max_lag = 0.5·max_dist)
        # — and that the Directional/Rose tabs already use — so the fitted curve overlays the
        # dots at any n_lags. variogram_panel's median-cutoff version binned differently.
        from utils import compute_empirical_variogram as _utils_vg
        from ui.variogram_panel import compute_model_curve
        canvas = self._vario_canvases.get("Omnidirectional")
        if canvas is None or self._ctrl._X is None:
            return
        is_gp = self._ctrl._engine == "gp"
        fig = canvas.fig; fig.clear(); ax = fig.add_subplot(111)
        ax.set_xlabel("Lag distance"); ax.set_ylabel("Semivariance")
        ax.set_title("Empirical Variogram" if is_gp else "Variogram Fit")
        n_lags = self._nlags_spin.value()
        emp = _utils_vg(self._ctrl._X, self._ctrl._y, n_lags=n_lags)
        valid = emp["n_pairs"] > 0
        lags, sv = emp["lags"][valid], emp["semivariance"][valid]
        if len(lags):
            ax.scatter(lags, sv, color="#1f77b4", s=28, zorder=3, label="Empirical")
        if is_gp:
            ax.text(0.5, 0.92,
                    "GP fits by marginal likelihood —\n"
                    "the kernel is not fitted to the empirical variogram.",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=9, fontstyle="italic", color="gray")
        elif len(lags):
            h = np.linspace(0, lags[-1] * 1.2, 200)
            gam = compute_model_curve(
                preset.get("model", "spherical"), h,
                preset.get("range", 300), preset.get("psill", 5.0),
                preset.get("nugget", 0.5), preset.get("alpha", 1.0))
            ax.plot(h, gam, color="#d62728", lw=2, label=preset.get("model", "?"))
            ax.legend(fontsize=9)
        canvas.draw_idle()

    def _redraw_directional(self):
        import matplotlib.pyplot as plt
        canvas = self._vario_canvases.get("Directional 15°")
        if canvas is None:
            return
        fig = canvas.fig; fig.clear(); ax = fig.add_subplot(111)
        if not self._dir_cache:
            ax.set_title("Directional Variograms (load data first)", fontsize=10)
            canvas.draw_idle()
            return
        cmap = plt.cm.tab20
        n = len(self._dir_cache)
        for i, dv in enumerate(self._dir_cache):
            valid = dv["n_pairs"] > 0
            ax.plot(dv["lags"][valid], dv["semivariance"][valid], "o-",
                    color=cmap(i / max(n, 1)), ms=4, lw=1.2,
                    label=f"{dv['direction']:.0f}°")
        ax.set_xlabel("Lag distance", fontsize=9)
        ax.set_ylabel("Semivariance", fontsize=9)
        ax.set_title("Directional Variograms (15° intervals)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        canvas.draw_idle()

    def _redraw_anisotropy_rose(self, preset):
        import numpy as np
        canvas = self._vario_canvases.get("Anisotropy Rose")
        if canvas is None:
            return
        fig = canvas.fig; fig.clear(); ax = fig.add_subplot(111, projection="polar")
        # Empirical range per direction from cached directional variograms
        if self._dir_cache:
            angles_deg, ranges = [], []
            for dv in self._dir_cache:
                valid = dv["n_pairs"] > 0
                lags_v = dv["lags"][valid]; sv_v = dv["semivariance"][valid]
                if len(lags_v) < 3:
                    continue
                target = 0.95 * float(np.max(sv_v))
                above = np.where(sv_v >= target)[0]
                est = float(lags_v[above[0]]) if len(above) else float(lags_v[-1])
                angles_deg.append(dv["direction"]); ranges.append(est)
            if angles_deg:
                a = np.array(angles_deg, float); r = np.array(ranges, float)
                a_full = np.deg2rad(np.concatenate([a, a + 180.0]))
                r_full = np.concatenate([r, r])
                order = np.argsort(a_full)
                a_full, r_full = a_full[order], r_full[order]
                a_full = np.append(a_full, a_full[0]); r_full = np.append(r_full, r_full[0])
                ax.plot(a_full, r_full, "o-", color="steelblue", ms=4, lw=1.5,
                        label="Empirical range")
                ax.fill(a_full, r_full, color="steelblue", alpha=0.15)
        # Fitted ellipse from current slider values
        r_major = max(self._sliders["Range"].value(), 1e-6)
        ratio = max(self._sliders["Anisotropy ×"].value(), 1.0)
        r_minor = r_major / ratio
        ang = np.deg2rad(self._sliders["Angle (°)"].value())
        theta = np.linspace(0, 2 * np.pi, 240)
        denom = np.sqrt((r_minor * np.cos(theta - ang)) ** 2
                        + (r_major * np.sin(theta - ang)) ** 2)
        r_ell = (r_major * r_minor) / np.where(denom == 0, 1e-9, denom)
        ax.plot(theta, r_ell, "r-", lw=2, label="Fitted ellipse")
        ax.set_title("Anisotropy Rose Diagram", fontsize=10, pad=14)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.28, 1.10))
        canvas.draw_idle()

    def _on_params_ready(self, params: dict):
        """Auto-fit completed — populate sliders + redraw variogram fit."""
        import numpy as np
        if "best_model" in params:  # kriging
            bm = params["best_model"]
            idx = self._krig_model_combo.findData(bm)
            if idx >= 0:
                self._krig_model_combo.setCurrentIndex(idx)
        if "range" in params:
            self._sliders["Range"].setValue(float(params["range"]))
        if "psill" in params:
            self._sliders["Sill (psill)"].setValue(float(params["psill"]))
        if "nugget" in params:
            self._sliders["Nugget"].setValue(float(params["nugget"]))
        if "rotation_angle_deg" in params:
            self._sliders["Angle (°)"].setValue(float(params["rotation_angle_deg"]))
        if "anisotropy_ratio" in params:
            self._sliders["Anisotropy ×"].setValue(
                min(float(params["anisotropy_ratio"]), 15.0))
        if "alpha" in params:
            self._sliders["Alpha"].setValue(float(params["alpha"]))
        # GP params
        if "kernel_type" in params:
            kt = params["kernel_type"]
            for i in range(self._gp_kernel_combo.count()):
                if self._gp_kernel_combo.itemText(i).startswith(kt):
                    self._gp_kernel_combo.setCurrentIndex(i)
                    break
        ls = params.get("length_scale")
        if ls is not None:
            if isinstance(ls, (list, tuple)) and len(ls):
                self._sliders["Range"].setValue(float(max(ls)))
        # Redraw with new params
        self._fire_sliders()

    # ── progress bar ───────────────────────────────────────────────────
    def _on_busy_started(self, label: str):
        # Indeterminate (busy) animation while a long op is in flight.
        self._progress.setRange(0, 0)
        self._progress.setFormat(label or "Working…")
        self._progress.setVisible(True)
        # Inline preview blocks the main thread; force a paint so the bar shows.
        QApplication.processEvents()

    def _on_busy_finished(self):
        self._progress.setVisible(False)
        self._progress.setRange(0, 100)
        self._progress.reset()

    def _on_progress(self, percent: int, label: str):
        # Determinate update parsed from the subprocess stage markers.
        if percent <= 0 and not label:
            self._progress.setVisible(False)
            self._progress.setRange(0, 100)
            self._progress.reset()
            return
        self._progress.setRange(0, 100)
        self._progress.setValue(int(percent))
        self._progress.setFormat(f"{label}  ({percent}%)" if label else f"{percent}%")
        self._progress.setVisible(True)

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
            # Also export variogram tab figures
            for tab_name, canvas in self._vario_canvases.items():
                p = Path(folder) / f"vario_{tab_name.replace(' ', '_').lower()}.png"
                canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
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
        self._sidebar.expandSection(0)

    def _restore_state(self):
        s = QSettings("InterpEngine", "GeospatialApp")
        geo = s.value("geometry")
        if geo:
            ok = self.restoreGeometry(geo)
            if not ok:
                self.resize(1400, 900)  # fallback if restored size doesn't fit

    def closeEvent(self, event):
        s = QSettings("InterpEngine", "GeospatialApp")
        s.setValue("geometry", self.saveGeometry())
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = GeospatialApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
