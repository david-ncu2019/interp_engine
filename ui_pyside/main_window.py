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
    QHBoxLayout,
    QCheckBox, QComboBox, QPushButton,
    QFileDialog, QDialog, QDialogButtonBox, QSpinBox, QDoubleSpinBox, QTabWidget,
    QProgressBar, QDockWidget,
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
        self._refreshing_dock = False  # re-entrancy guard for _on_dock_visibility

        splitter = QSplitter(Qt.Horizontal)
        self._sidebar = self._build_sidebar()
        splitter.addWidget(self._sidebar)
        self._plot_area = self._build_plot_grid()
        splitter.addWidget(self._plot_area)
        splitter.setSizes([320, 1080])
        self.setCentralWidget(splitter)

        # Capture the pristine 2×2 dock arrangement for Reset Layout.
        self._default_dock_state = self._plot_area.saveState()

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
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(["Ordinary Kriging", "Gaussian Process"])
        self._engine_combo.setToolTip(
            "Interpolation method. Ordinary Kriging = classical geostatistics\n"
            "(fast, fits a variogram model). Gaussian Process = probabilistic\n"
            "ML approach (slower, full predictive distribution).")
        el.addWidget(self._engine_combo)
        dl.addWidget(eng_w)

        # Grid mode toggle
        grid_mode_w = QWidget(); gml = QVBoxLayout(grid_mode_w)
        gml.setContentsMargins(0, 6, 0, 0)
        gml.addWidget(QLabel("Prediction grid:"))
        self._grid_mode_combo = QComboBox()
        self._grid_mode_combo.addItems(["Auto (convex hull)", "Custom points (CSV)"])
        self._grid_mode_combo.setToolTip(
            "Auto = regular grid bounded by the convex hull of your data.\n"
            "Custom = predict at specific X,Y locations from a CSV file\n"
            "(e.g., monitoring stations or well locations).")
        gml.addWidget(self._grid_mode_combo)

        # Custom points file picker (hidden until custom mode selected)
        self._custom_pts_picker = FilePicker()
        self._custom_pts_picker.setVisible(False)
        gml.addWidget(self._custom_pts_picker)
        self._custom_pts_col_x_label = QLabel("X column:"); self._custom_pts_col_x_label.setVisible(False)
        self._custom_pts_col_x = QComboBox(); self._custom_pts_col_x.setVisible(False)
        self._custom_pts_col_y_label = QLabel("Y column:"); self._custom_pts_col_y_label.setVisible(False)
        self._custom_pts_col_y = QComboBox(); self._custom_pts_col_y.setVisible(False)
        gml.addWidget(self._custom_pts_col_x_label); gml.addWidget(self._custom_pts_col_x)
        gml.addWidget(self._custom_pts_col_y_label); gml.addWidget(self._custom_pts_col_y)
        dl.addWidget(grid_mode_w)

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

        # Lag distance spinbox = lag separation / bin width (0 = auto)
        self._max_lag_label = QLabel("Lag distance:")
        self._max_lag_spin = QDoubleSpinBox()
        self._max_lag_spin.setRange(0, 99999)
        self._max_lag_spin.setValue(0)
        self._max_lag_spin.setDecimals(1)
        self._max_lag_spin.setSpecialValueText("auto")
        self._max_lag_spin.setToolTip(
            "Lag separation — the width of each variogram bin.\n"
            "Total reach = number of lags × this value.\n"
            "'auto' (0) derives a good value from your data spacing.")
        self._max_lag_spin.valueChanged.connect(self._fire_sliders)
        vl.addWidget(self._max_lag_label)
        vl.addWidget(self._max_lag_spin)

        # Lag tolerance spinbox = ± window around each lag (0 = auto = half lag distance)
        self._lag_tol_label = QLabel("Lag tolerance:")
        self._lag_tol_spin = QDoubleSpinBox()
        self._lag_tol_spin.setRange(0, 99999)
        self._lag_tol_spin.setValue(0)
        self._lag_tol_spin.setDecimals(1)
        self._lag_tol_spin.setSpecialValueText("auto")
        self._lag_tol_spin.setToolTip(
            "± window around each lag center when grouping point pairs.\n"
            "'auto' (0) uses half the lag distance (contiguous bins).\n"
            "Larger = overlapping bins (smoother variogram); smaller =\n"
            "gaps between bins.")
        self._lag_tol_spin.valueChanged.connect(self._fire_sliders)
        vl.addWidget(self._lag_tol_label)
        vl.addWidget(self._lag_tol_spin)

        # Lock checkboxes for lag parameters
        lock_row = QWidget(); lrl = QHBoxLayout(lock_row); lrl.setContentsMargins(0, 2, 0, 0)
        self._lock_nlags_cb = QCheckBox("Lock n_lags")
        self._lock_nlags_cb.setChecked(True)
        self._lock_nlags_cb.setToolTip(
            "When locked: 'Optimize Parameters' keeps n_lags at its\n"
            "current value. When unlocked: the optimizer searches for\n"
            "the best n_lags (along with unlocked lag distance).")
        self._lock_lag_cb = QCheckBox("Lock lag distance")
        self._lock_lag_cb.setChecked(False)
        self._lock_lag_cb.setToolTip(
            "When locked: 'Optimize Parameters' keeps lag distance at\n"
            "its current value. When unlocked: the optimizer searches for\n"
            "the best lag distance (along with unlocked n_lags).")
        lrl.addWidget(self._lock_nlags_cb); lrl.addWidget(self._lock_lag_cb)
        vl.addWidget(lock_row)

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

        # Run Options — cross-validation toggle (default OFF = interpolation only)
        run_opts = QWidget(); rol = QVBoxLayout(run_opts)
        rol.setContentsMargins(0, 6, 0, 0)
        rol.addWidget(QLabel("Run Options:"))
        self._cv_cb = QCheckBox("Compute cross-validation (slower)")
        self._cv_cb.setChecked(False)
        self._cv_cb.setToolTip(
            "Leave-one-out / k-fold CV gives accuracy metrics (MAE/RMSE/R²)\n"
            "but is O(N³) and slow for large datasets.\n"
            "Off = interpolation only (no CV dashboard, no metrics).")
        rol.addWidget(self._cv_cb)
        vl.addWidget(run_opts)

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

        # Save / Load config buttons
        save_load_row = QWidget(); slrl = QHBoxLayout(save_load_row)
        slrl.setContentsMargins(0, 2, 0, 0)
        self._save_cfg_btn = QPushButton("💾 Save Config…")
        self._save_cfg_btn.setToolTip(
            "Save current model, engine, and slider settings as a YAML config file.\n"
            "You can reload it later or use it with the CLI: python main.py saved.yaml")
        self._load_cfg_btn = QPushButton("📂 Load Config…")
        self._load_cfg_btn.setToolTip(
            "Load a saved YAML config file. This will restore engine choice,\n"
            "variogram model, slider values, and preprocessing settings.")
        slrl.addWidget(self._save_cfg_btn); slrl.addWidget(self._load_cfg_btn)
        vl.addWidget(save_load_row)
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

        # Section 5: Validation (ground truth)
        sec_val = sb.addSection("Validation")
        vw2 = QWidget(); vl2 = QVBoxLayout(vw2); vl2.setContentsMargins(4, 4, 4, 4)
        vl2.addWidget(QLabel("Ground truth file:"))
        self._gt_picker = FilePicker()
        vl2.addWidget(self._gt_picker)
        vl2.addWidget(QLabel("Value column:"))
        self._gt_col_cb = QComboBox()
        self._gt_col_cb.setToolTip(
            "The column in the ground truth CSV containing the\n"
            "observed values to compare against predictions.")
        vl2.addWidget(self._gt_col_cb)
        self._gt_compare_btn = QPushButton("Compare")
        self._gt_compare_btn.setToolTip(
            "Fit the model with current slider parameters, predict at the\n"
            "ground truth locations, and open a validation window with\n"
            "scatter plots, error maps, metrics, and residual diagnostics.")
        self._gt_compare_btn.setEnabled(False)
        vl2.addWidget(self._gt_compare_btn)
        sec_val.setContent(vw2)

        # Section 6: Log (collapsed)
        sec_log = sb.addSection("Log")
        self._log = LogConsole()
        sec_log.setContent(self._log)

        return sb

    # ── plot grid ────────────────────────────────────────────────────
    def _build_plot_grid(self) -> QMainWindow:
        """Build the right pane as a nested QMainWindow of detachable docks.

        Each of the four panels is wrapped in a QDockWidget (Movable | Floatable,
        NOT Closable so panels can't be lost). The PlotPanel / MplCanvas objects
        themselves are stored in self._plots / self._vario_canvases exactly as
        before, so all redraw code keeps targeting the same live canvases even
        when a dock is floated onto another monitor.

        Using a nested QMainWindow + addDockWidget/splitDockWidget avoids the
        QDockWidget-in-QGridLayout pitfall (commit f76295d) that hid content.
        """
        area = QMainWindow()
        area.setObjectName("plotAreaMainWindow")
        # A zero-size central widget so the docks fill the entire area
        # (otherwise an empty central widget eats space between the docks).
        central = QWidget()
        central.setMaximumSize(0, 0)
        area.setCentralWidget(central)

        self._plots = {}
        self._docks = {}

        # Create the three PlotPanels and the variogram tab widget — SAME
        # objects the redraw code references via self._plots / self._vario_tabs.
        pred = PlotPanel(title="Prediction Surface")
        self._plots["Prediction Surface"] = pred
        uncert = PlotPanel(title="Uncertainty (std)")
        self._plots["Uncertainty (std)"] = uncert
        cv = PlotPanel(title="CV Dashboard")
        self._plots["CV Dashboard"] = cv
        self._vario_tabs = self._build_vario_tabs()

        def _mk_dock(name: str, widget: QWidget, obj: str) -> QDockWidget:
            d = QDockWidget(name, area)
            d.setObjectName(obj)          # REQUIRED for saveState/restoreState
            d.setWidget(widget)
            d.setFeatures(QDockWidget.DockWidgetMovable
                          | QDockWidget.DockWidgetFloatable
                          | QDockWidget.DockWidgetClosable)
            self._docks[name] = d
            return d

        dock_pred   = _mk_dock("Prediction Surface", pred, "dockPrediction")
        dock_uncert = _mk_dock("Uncertainty (std)", uncert, "dockUncertainty")
        dock_vario  = _mk_dock("Variogram", self._vario_tabs, "dockVariogram")
        dock_cv     = _mk_dock("CV Dashboard", cv, "dockCV")

        # 2×2 arrangement:
        #   [ Prediction | Uncertainty ]
        #   [ Variogram  |  CV Dashboard ]
        area.addDockWidget(Qt.TopDockWidgetArea, dock_pred)
        area.splitDockWidget(dock_pred, dock_uncert, Qt.Horizontal)
        area.splitDockWidget(dock_pred, dock_vario, Qt.Vertical)
        area.splitDockWidget(dock_uncert, dock_cv, Qt.Vertical)

        # When a closed dock is re-shown, repaint its canvas from cached state so
        # it isn't blank/stale. Default-arg capture binds the name per iteration.
        for _nm, _dk in self._docks.items():
            _dk.visibilityChanged.connect(
                lambda visible, nm=_nm: self._on_dock_visibility(nm, visible))
        return area

    def _build_vario_tabs(self) -> QTabWidget:
        from ui_pyside.mpl_canvas import MplCanvas
        tabs = QTabWidget()
        for tab_name in ["Omnidirectional", "Directional 15°", "Anisotropy Rose"]:
            canvas = MplCanvas(figsize=(5.0, 3.5), dpi=96)
            self._vario_canvases[tab_name] = canvas
            tabs.addTab(canvas, tab_name)
        return tabs

    def _on_dock_visibility(self, name: str, visible: bool):
        """When a dock is re-shown, repaint its canvas from cached data so it
        isn't blank/stale. No-op if data isn't ready or we're mid-refresh."""
        if not visible or self._refreshing_dock:
            return
        if not hasattr(self, "_plots"):      # before widgets exist
            return
        self._refreshing_dock = True
        try:
            last = self._ctrl._last_full
            if name in ("Prediction Surface", "Uncertainty (std)"):
                grid = last.get("grid") if last else None
                if grid is not None and "mean" in grid:
                    self._draw_surface(grid)
            elif name == "CV Dashboard":
                cv_df = last.get("cv_df") if last else None
                if cv_df is not None:
                    self._draw_cv(cv_df)
                else:
                    self._show_cv_placeholder()
            elif name == "Variogram":
                if self._ctrl._X is not None:
                    self._draw_variogram_tabs()
            # No data yet → leave the existing placeholder untouched (no error).
        except Exception:
            pass
        finally:
            self._refreshing_dock = False

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
        panels_menu = view_menu.addMenu("&Panels")
        # Canonical Qt reopen mechanism: checkable toggle action per dock.
        # Auto-titled from the dock window title, auto-syncs with the ✕ button.
        for _name in ("Prediction Surface", "Uncertainty (std)",
                      "Variogram", "CV Dashboard"):
            _dock = self._docks.get(_name)
            if _dock is not None:
                panels_menu.addAction(_dock.toggleViewAction())
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
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self._krig_model_combo.currentTextChanged.connect(self._fire_sliders)
        self._gp_kernel_combo.currentTextChanged.connect(self._fire_sliders)
        for sl in self._sliders.values():
            sl.valueChanged.connect(lambda v: self._fire_sliders())
        self._live_cb.toggled.connect(c.set_live)
        self._cv_cb.toggled.connect(c.set_compute_cv)
        self._cv_cb.toggled.connect(self._on_cv_toggled)
        self._run_btn.clicked.connect(c.run_full)
        self._auto_btn.clicked.connect(c.auto_fit)
        self._export_btn.clicked.connect(self._export)
        self._detrend_cb.toggled.connect(self._on_preproc_changed)
        self._detrend_order_cb.currentIndexChanged.connect(self._on_preproc_changed)
        self._detrend_auto_cb.toggled.connect(self._on_preproc_changed)
        self._nst_combo.currentIndexChanged.connect(self._on_preproc_changed)
        self._lock_nlags_cb.toggled.connect(self._on_lag_lock_changed)
        self._lock_lag_cb.toggled.connect(self._on_lag_lock_changed)
        self._grid_mode_combo.currentIndexChanged.connect(self._on_grid_mode_changed)
        self._custom_pts_picker.fileSelected.connect(self._on_custom_pts_file)
        self._custom_pts_col_x.currentTextChanged.connect(
            lambda: self._on_custom_pts_cols_changed())
        self._custom_pts_col_y.currentTextChanged.connect(
            lambda: self._on_custom_pts_cols_changed())
        self._save_cfg_btn.clicked.connect(self._save_config)
        self._load_cfg_btn.clicked.connect(self._load_config)
        self._gt_picker.fileSelected.connect(self._on_gt_file_selected)
        self._gt_compare_btn.clicked.connect(self._on_validate)
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

    def _on_engine_changed(self, idx):
        mode = "gp" if idx == 1 else "kriging"
        self._ctrl.set_engine(mode)
        is_krig = mode == "kriging"
        self._krig_model_label.setVisible(is_krig)
        self._krig_model_combo.setVisible(is_krig)
        self._nlags_label.setVisible(is_krig)
        self._nlags_spin.setVisible(is_krig)
        self._max_lag_label.setVisible(is_krig)
        self._max_lag_spin.setVisible(is_krig)
        self._lag_tol_label.setVisible(is_krig)
        self._lag_tol_spin.setVisible(is_krig)
        self._lock_nlags_cb.setVisible(is_krig)
        self._lock_lag_cb.setVisible(is_krig)
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

    def _on_lag_lock_changed(self, *_):
        self._ctrl._state["kriging_lock_n_lags"] = self._lock_nlags_cb.isChecked()
        self._ctrl._state["kriging_lock_max_lag"] = self._lock_lag_cb.isChecked()

    def _on_grid_mode_changed(self, idx):
        use_custom = idx == 1
        self._custom_pts_picker.setVisible(use_custom)
        self._custom_pts_col_x_label.setVisible(use_custom)
        self._custom_pts_col_x.setVisible(use_custom)
        self._custom_pts_col_y_label.setVisible(use_custom)
        self._custom_pts_col_y.setVisible(use_custom)
        self._ctrl.set_grid_mode(use_custom)
        if use_custom:
            self._live_cb.setChecked(False)
            self._status_label.setText("Custom points mode — live preview disabled.")

    def _on_custom_pts_file(self, path):
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            headers = csv.DictReader(f).fieldnames or []
        for cb in (self._custom_pts_col_x, self._custom_pts_col_y):
            cb.clear(); cb.addItems(headers)

    def _on_custom_pts_cols_changed(self):
        cx = self._custom_pts_col_x.currentText()
        cy = self._custom_pts_col_y.currentText()
        fp = self._custom_pts_picker.path()
        if fp and cx and cy:
            self._ctrl.set_prediction_points_file(fp, cx, cy)

    def _save_config(self):
        from pathlib import Path
        import yaml
        cfg = self._ctrl.build_config_from_current_state()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", str(Path.home() / "interp_config.yaml"),
            "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        self._status_label.setText(f"Config saved to {Path(path).name}")

    def _load_config(self):
        from pathlib import Path
        import yaml
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", str(Path.home()),
            "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        with open(path) as f:
            cfg = yaml.safe_load(f)
        updates = self._ctrl.apply_config(cfg)

        # Apply engine mode
        if updates.get("engine_mode") == "gp":
            self._engine_combo.setCurrentIndex(1)
        else:
            self._engine_combo.setCurrentIndex(0)

        # Apply preprocessing controls
        self._detrend_cb.setChecked(updates.get("detrend_enabled", False))
        order_idx = max(0, min(2, updates.get("detrend_order", 1) - 1))
        self._detrend_order_cb.setCurrentIndex(order_idx)
        self._detrend_auto_cb.setChecked(updates.get("detrend_auto", False))
        nst_map = {False: 0, True: 1, None: 2}
        self._nst_combo.setCurrentIndex(nst_map.get(updates.get("nst_enabled"), 0))

        # Apply model/kernel
        if updates.get("kriging_model"):
            idx = self._krig_model_combo.findData(updates["kriging_model"])
            if idx >= 0:
                self._krig_model_combo.setCurrentIndex(idx)
        if updates.get("gp_kernel"):
            for i in range(self._gp_kernel_combo.count()):
                if self._gp_kernel_combo.itemText(i).startswith(updates["gp_kernel"]):
                    self._gp_kernel_combo.setCurrentIndex(i)
                    break

        # Apply n_lags + GSLIB lag binning (lag distance / tolerance)
        if "n_lags" in updates:
            self._nlags_spin.setValue(updates["n_lags"])
        if "lag_width" in updates:
            self._max_lag_spin.setValue(float(updates["lag_width"] or 0.0))
        if "lag_tolerance" in updates:
            self._lag_tol_spin.setValue(float(updates["lag_tolerance"] or 0.0))

        # Apply compute-CV toggle
        self._cv_cb.setChecked(bool(updates.get("compute_cv", False)))

        # Apply sliders
        s = updates.get("sliders", {})
        for name, val in s.items():
            if name in self._sliders:
                self._sliders[name].setValue(val)

        self._status_label.setText(f"Config loaded from {Path(path).name}")
        self._fire_sliders()

    def _on_gt_file_selected(self, path):
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            headers = csv.DictReader(f).fieldnames or []
        self._gt_col_cb.clear(); self._gt_col_cb.addItems(headers)
        # Auto-select a likely value column
        for guess in ("Value", "value", "Z", "elev", "elevation", "grade"):
            if guess in headers:
                self._gt_col_cb.setCurrentText(guess)
                break
        self._gt_compare_btn.setEnabled(len(headers) > 0)

    def _on_validate(self):
        fp = self._gt_picker.path()
        col = self._gt_col_cb.currentText()
        if not fp or not col:
            return
        self._status_label.setText("Running ground truth comparison…")
        # The comparison fits + predicts on the main thread and can be slow for a
        # large ground-truth grid; show the busy indicator so the UI isn't a silent freeze.
        self._on_busy_started("Comparing to ground truth…")
        try:
            result = self._ctrl.compare_ground_truth(fp, col)
            from ui_pyside.ground_truth_window import GroundTruthWindow
            win = GroundTruthWindow(
                result["metrics"], result["residuals"],
                result["gt_obs"], result["gt_pred"],
                result["gt_X"], result["gt_Y"],
                result["gt_col"], parent=self)
            win.exec()
            self._status_label.setText(
                f"Ground truth: MAE={result['metrics']['mae']:.3f}, "
                f"RMSE={result['metrics']['rmse']:.3f}, "
                f"R²={result['metrics']['r2']:.3f}")
        except Exception as exc:
            self._status_label.setText(f"Validation failed: {exc}")
        finally:
            self._on_busy_finished()

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
        self._ctrl._state["kriging_lag_width"] = self._max_lag_spin.value()
        self._ctrl._state["kriging_lag_tol"] = self._lag_tol_spin.value()
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

    def _on_cv_toggled(self, checked: bool):
        """When CV is switched OFF, make the off-state obvious (placeholder +
        metrics hint) instead of leaving stale numbers around."""
        if not checked:
            self._show_cv_placeholder()
            self._metrics_label.setText("CV off — enable to see MAE/RMSE/R²")

    def _show_cv_placeholder(self):
        dp = self._plots.get("CV Dashboard")
        if dp is None:
            return
        fig = dp.canvas.fig; fig.clear(); ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Cross-validation not computed.\n"
                "Enable 'Compute cross-validation' and re-run.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        dp.canvas.draw_idle()

    def _on_result(self, result):
        if result.get("preview"):
            # Live preview: only the surface/uncertainty panels update.
            grid = result.get("grid")
            if grid is not None and "mean" in grid:
                self._draw_surface(grid)
            return
        grid = result.get("grid")
        if grid is not None and "mean" in grid:
            self._draw_surface(grid)
        cv_df = result.get("cv_df")
        if cv_df is not None:
            self._draw_cv(cv_df)
        else:
            # Full run with CV off → show the explanation, not a stale dashboard.
            self._show_cv_placeholder()

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
    def _lag_binning(self):
        """Current GSLIB-style lag params from the sidebar (0 spinbox = auto/None)."""
        lw = self._max_lag_spin.value()
        lt = self._lag_tol_spin.value()
        return {
            "n_lags": self._nlags_spin.value(),
            "lag_width": lw if lw > 0 else None,
            "lag_tolerance": lt if lt > 0 else None,
        }

    def _compute_directional(self):
        """Compute 12 directional empirical variograms (15° intervals) and cache."""
        if self._ctrl._X is None:
            self._dir_cache = None
            return
        from utils import compute_empirical_variogram as utils_variogram
        directions = list(range(0, 180, 15))  # 0, 15, …, 165
        try:
            self._dir_cache = utils_variogram(
                self._ctrl._X, self._ctrl._y,
                directions=directions, **self._lag_binning())
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
        emp = _utils_vg(self._ctrl._X, self._ctrl._y, **self._lag_binning())
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

        # Directional variograms (faded)
        cmap = plt.cm.tab20
        n = len(self._dir_cache)
        for i, dv in enumerate(self._dir_cache):
            valid = dv["n_pairs"] > 0
            ax.plot(dv["lags"][valid], dv["semivariance"][valid], "o-",
                    color=cmap(i / max(n, 1)), ms=4, lw=1.2,
                    alpha=0.75, label=f"{dv['direction']:.0f}°")

        # Omnidirectional variogram (prominent overlay — black diamonds)
        from utils import compute_empirical_variogram as _utils_vg
        n_lags = self._nlags_spin.value()
        omni = _utils_vg(self._ctrl._X, self._ctrl._y, n_lags=n_lags)
        om_valid = omni["n_pairs"] > 0
        if om_valid.any():
            ax.plot(omni["lags"][om_valid], omni["semivariance"][om_valid],
                    "D-", color="black", ms=6, lw=1.5, alpha=1.0,
                    label="Omnidirectional", zorder=10)

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
        # Surface the lag binning the optimizer chose (kriging only). Block signals so
        # the trailing _fire_sliders() does a single redraw instead of three.
        for spin, key, cast in ((self._nlags_spin, "n_lags", int),
                                (self._max_lag_spin, "lag_width", float),
                                (self._lag_tol_spin, "lag_tolerance", float)):
            if key in params:
                spin.blockSignals(True)
                spin.setValue(cast(params[key]))
                spin.blockSignals(False)
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
        dlg.setWindowTitle("Export Results")
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)

        # ── Figures section ──
        figs_label = QLabel("☐ Figures")
        figs_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(figs_label)
        fig_cbs = {}
        fig_names = [
            ("fig_pred", "Prediction Surface"),
            ("fig_uncert", "Uncertainty (std)"),
            ("fig_cv", "CV Dashboard"),
            ("fig_vario_omni", "Variogram — Omnidirectional"),
            ("fig_vario_dir", "Variogram — Directional 15°"),
            ("fig_vario_rose", "Variogram — Anisotropy Rose"),
        ]
        for key, label in fig_names:
            cb = QCheckBox("  " + label); cb.setChecked(True)
            fig_cbs[key] = cb; layout.addWidget(cb)

        # ── Data section ──
        data_label = QLabel("☐ Data")
        data_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(data_label)
        data_cbs = {}
        data_items = [
            ("data_grid", "Predicted grid (.npz)"),
            ("data_cv", "CV results (.csv)"),
            ("data_params", "Parameters (.yaml)"),
        ]
        for key, label in data_items:
            cb = QCheckBox("  " + label); cb.setChecked(True)
            data_cbs[key] = cb; layout.addWidget(cb)

        # ── Validation section (only if GT was run) ──
        has_gt = self._ctrl._gt_result is not None
        val_label = QLabel("☐ Validation")
        val_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(val_label)
        val_cbs = {}
        val_items = [
            ("val_metrics", "Ground truth metrics (.csv)"),
            ("val_fig", "Comparison figure (.png)"),
        ]
        for key, label in val_items:
            cb = QCheckBox("  " + label)
            cb.setChecked(has_gt); cb.setEnabled(has_gt)
            val_cbs[key] = cb; layout.addWidget(cb)

        # ── Select All / None ──
        sel_row = QWidget(); srl = QHBoxLayout(sel_row); srl.setContentsMargins(0, 4, 0, 0)
        all_btn = QPushButton("Select All")
        none_btn = QPushButton("Deselect All")
        srl.addWidget(all_btn); srl.addWidget(none_btn)
        layout.addWidget(sel_row)

        def _select_all(checked):
            for d in (fig_cbs, data_cbs, val_cbs):
                for cb in d.values():
                    if cb.isEnabled():
                        cb.setChecked(checked)
        all_btn.clicked.connect(lambda: _select_all(True))
        none_btn.clicked.connect(lambda: _select_all(False))

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
        # Figures
        if any(cb.isChecked() for cb in fig_cbs.values()):
            plot_map = {
                "fig_pred": "Prediction Surface",
                "fig_uncert": "Uncertainty (std)",
            }
            for key, name in plot_map.items():
                if fig_cbs.get(key) and fig_cbs[key].isChecked():
                    pp = self._plots.get(name)
                    if pp:
                        p = Path(folder) / f"{name.replace(' ', '_').lower()}.png"
                        pp.canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
                        saved.append(p.name)
            if fig_cbs.get("fig_cv") and fig_cbs["fig_cv"].isChecked():
                pp = self._plots.get("CV Dashboard")
                if pp:
                    p = Path(folder) / "cv_dashboard.png"
                    pp.canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
                    saved.append(p.name)
            vario_map = {
                "fig_vario_omni": "Omnidirectional",
                "fig_vario_dir": "Directional 15°",
                "fig_vario_rose": "Anisotropy Rose",
            }
            for key, tab_name in vario_map.items():
                if fig_cbs.get(key) and fig_cbs[key].isChecked():
                    canvas = self._vario_canvases.get(tab_name)
                    if canvas:
                        p = Path(folder) / f"vario_{tab_name.replace(' ', '_').lower()}.png"
                        canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
                        saved.append(p.name)

        # Data
        want_grid = data_cbs.get("data_grid") and data_cbs["data_grid"].isChecked()
        want_cv = data_cbs.get("data_cv") and data_cbs["data_cv"].isChecked()
        want_params = data_cbs.get("data_params") and data_cbs["data_params"].isChecked()
        ctrl_saved = self._ctrl.export(folder, want_grid, want_cv)
        if ctrl_saved:
            saved.extend(ctrl_saved)
        if want_params:
            import yaml
            cfg = self._ctrl.build_config_from_current_state()
            p = Path(folder) / "parameters.yaml"
            with open(p, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            saved.append(p.name)

        # Validation
        if has_gt:
            if val_cbs.get("val_metrics") and val_cbs["val_metrics"].isChecked():
                import csv
                p = Path(folder) / "ground_truth_metrics.csv"
                m = self._ctrl._gt_result["metrics"]
                with open(p, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=m.keys())
                    writer.writeheader(); writer.writerow(m)
                saved.append(p.name)
            if val_cbs.get("val_fig") and val_cbs["val_fig"].isChecked():
                # Regenerate comparison figure from cached GT data
                import matplotlib.pyplot as plt
                import numpy as np
                gt = self._ctrl._gt_result
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                # Scatter
                ax = axes[0, 0]
                ax.scatter(gt["gt_obs"], gt["gt_pred"], alpha=0.6, s=36,
                           color="#1f77b4", edgecolors="white", linewidth=0.5)
                lims = [min(gt["gt_obs"].min(), gt["gt_pred"].min()),
                        max(gt["gt_obs"].max(), gt["gt_pred"].max())]
                pad = (lims[1] - lims[0]) * 0.05
                ax.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad],
                        "r--", lw=1.5, alpha=0.7)
                ax.set_xlabel(f"Observed ({gt.get('gt_col', 'Value')})")
                ax.set_ylabel("Predicted")
                ax.set_title("Predicted vs Observed")
                # Error map
                ax2 = axes[0, 1]
                res = gt["residuals"]
                abs_res = np.abs(res)
                max_abs = max(abs_res.max(), 1e-9)
                sizes = 20 + 120 * (abs_res / max_abs)
                colors = np.where(res >= 0, "#d62728", "#1f77b4")
                ax2.scatter(gt["gt_X"], gt["gt_Y"], c=colors, s=sizes,
                            alpha=0.75, edgecolors="white", linewidth=0.4)
                ax2.set_xlabel("X"); ax2.set_ylabel("Y")
                ax2.set_title("Spatial Error Map")
                ax2.set_aspect("equal")
                # Histogram
                ax3 = axes[1, 1]
                ax3.hist(res, bins=max(12, int(len(res)**0.5)), density=True,
                         color="steelblue", alpha=0.6, edgecolor="white")
                ax3.axvline(0, color="gray", ls="--", lw=1, alpha=0.5)
                ax3.set_xlabel("Residual"); ax3.set_title("Residual Distribution")
                fig.tight_layout()
                p = Path(folder) / "ground_truth_comparison.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(p.name)

        QMessageBox.information(self, "Export complete",
            "Saved:\n" + "\n".join(sorted(saved)) if saved else "Nothing selected.")

    def _about(self):
        QMessageBox.about(self, "About",
            "Interpolation Engine\nGeostatistics · Kriging · Gaussian Process")

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Set output folder")
        if d:
            self._ctrl._state["output_dir"] = d

    def _reset_layout(self):
        self._sidebar.expandSection(0)
        # Restore the pristine 2×2 dock arrangement (un-float / re-dock panels).
        try:
            self._plot_area.restoreState(self._default_dock_state)
        except Exception:
            pass
        # Guarantee every panel is visible again. restoreState of the pristine
        # snapshot already shows all four (it was captured with all visible), but
        # make it explicit so a closed panel is always recovered.
        for _dk in getattr(self, "_docks", {}).values():
            _dk.show()
            act = _dk.toggleViewAction()
            if not act.isChecked():
                act.setChecked(True)

    def _restore_state(self):
        s = QSettings("InterpEngine", "GeospatialApp")
        geo = s.value("geometry")
        if geo:
            ok = self.restoreGeometry(geo)
            if not ok:
                self.resize(1400, 900)  # fallback if restored size doesn't fit
        # Restore dock layout (floating/positions). Guarded so a corrupt or
        # stale blob from an older version can't crash startup.
        dock_state = s.value("dock_state")
        if dock_state:
            try:
                self._plot_area.restoreState(dock_state)
            except Exception:
                pass

    def closeEvent(self, event):
        s = QSettings("InterpEngine", "GeospatialApp")
        s.setValue("geometry", self.saveGeometry())
        try:
            s.setValue("dock_state", self._plot_area.saveState())
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = GeospatialApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
