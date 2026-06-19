# PySide6 Minimalist UI Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the interpolation engine GUI with PySide6 + QDarkStyle + matplotlib, featuring an accordion sidebar and detachable 2x2 plot grid for multi-monitor use.

**Architecture:** All new UI in `ui_pyside/`. Backend computation (`src/`, `utils.py`, `ui/live_predictor.py`, `ui/engine_runner.py`, `main.py`) reused with zero code changes. `WorkspaceController(QObject)` is a signal/slot mediator between widgets. Plots render via `FigureCanvasQTAgg` (direct matplotlib Qt backend). QProcess replaces threading.Thread+subprocess.Popen for engine runs.

**Tech Stack:** Python 3.12, PySide6 6.11, matplotlib 3.10, QDarkStyle, numpy. Conda env `fafalab2`. Spec: `docs/superpowers/specs/2026-06-19-pyside6-minimalist-redesign.md`.

## Global Constraints

- All new files in `ui_pyside/` (Python package). No Tkinter imports anywhere in the package.
- Existing `ui/`, `src/`, `utils.py`, `main.py` — **zero changes**. The Tkinter app stays intact. Both UIs coexist via different entry points.
- `ui_pyside/__init__.py` must harden DLL resolution before anything else imports (Python 3.8+ `os.add_dll_directory`; the diagnosed `gemini_env` PATH contamination from WS-4c).
- QDarkStyle must be installed (`pip install QDarkStyle`).
- No test framework — verify with: `py_compile` syntax check + `conda run -n fafalab2 python <script>` import/AST probe + user smoke test (`conda run -n fafalab2 python ui_pyside/main_window.py`). Per-task commit, user smoke test per task.
- Env python: `D:/Programs/miniconda3/Library/envs/fafalab2/python.exe`. Write temp `.py` scripts (avoid `python -c` multi-statement on Windows).

---

### Task 1: Package scaffold + DLL safety + QDarkStyle install

**Files:**
- Create: `ui_pyside/__init__.py`

- [ ] **Step 1: Install QDarkStyle**

```bash
conda run -n fafalab2 pip install QDarkStyle
```
Expected: `Successfully installed QDarkStyle-x.x.x`.

- [ ] **Step 2: Create `ui_pyside/__init__.py` with DLL hardening**

```python
"""
ui_pyside — PySide6 replacement UI for the interpolation engine.

Imports of this package are safe: DLL resolution is hardened before
any Qt or numerical library loads, preventing the gemini_env PATH
contamination that causes 0xc06d007f crashes on this Windows stack.
"""
import os as _os
import sys as _sys

# ── DLL safety: force fafalab2 native libs to the front ─────────────────────
_prefix = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in (r"Library\bin", r"Library\lib", r"DLLs", ""):
    _cand = _os.path.join(_sys.prefix, _d)
    if _os.path.isdir(_cand):
        try:
            _os.add_dll_directory(_cand)
        except (OSError, AttributeError):
            pass
    if _cand and _cand not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = _cand + ";" + _os.environ.get("PATH", "")

# Qt API — must be set before any Qt import
_os.environ.setdefault("QT_API", "pyside6")
```

- [ ] **Step 3: Verify DLL hardening works**

Write and run a temp probe that imports `ui_pyside` then does a bare numpy SVD (the operation that crashed before due to PATH contamination):

```python
import sys; sys.path.insert(0, r"D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine")
import ui_pyside   # must be FIRST import — triggers DLL hardening
import numpy as np
A = np.random.rand(500, 6)
u, s, v = np.linalg.svd(A, full_matrices=False)
print("SVD OK — DLL safety working")
```
Run: `conda run -n fafalab2 python _probe_dll.py` (as temp file, then `rm`)
Expected: `SVD OK — DLL safety working` (no crash).

- [ ] **Step 4: Verify PySide6 imports work through the hardened init**

Write and run:
```python
import sys; sys.path.insert(0, r"D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine")
import ui_pyside
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
app = QApplication.instance() or QApplication([])
w = QMainWindow(); w.setCentralWidget(QLabel("Qt works")); w.show()
print("PySide6 import OK, QMainWindow created")
app.quit()
```
Run: `conda run -n fafalab2 python _probe_qt.py` (temp file, then `rm`)
Expected: prints OK, no crash.

- [ ] **Step 5: Commit**

```bash
git add ui_pyside/__init__.py
git commit -m "feat: ui_pyside scaffold with DLL safety hardening

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: AnimatedSlider — port of LabeledSlider to Qt

**Files:**
- Create: `ui_pyside/animated_slider.py`

**Interfaces:**
- Produces: `class AnimatedSlider(QWidget)`
  - `__init__(self, parent, label: str, min_val: float, max_val: float, default: float)`
  - `value() -> float`
  - `setValue(float)`  (Qt style — camelCase for Qt API consistency)
  - `setRange(float, float)`
  - Signal: `valueChanged(float)`  (emitted after 100ms debounce)

- [ ] **Step 1: Create `ui_pyside/animated_slider.py`**

```python
"""AnimatedSlider — QSlider + QDoubleSpinBox + QLabel, debounced."""
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLabel,
                                QSlider, QDoubleSpinBox)
from PySide6.QtCore import Qt, Signal, QTimer


class AnimatedSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, parent=None, label="", min_val=0.0, max_val=1.0,
                 default=0.5):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._updating = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(100)
        self._debounce.timeout.connect(self._emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setFixedWidth(110)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(self._to_slider(default))
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(min_val, max_val)
        self._spin.setDecimals(3)
        self._spin.setValue(default)
        self._spin.setFixedWidth(80)
        self._spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self._spin)

    def _to_slider(self, val: float) -> int:
        return int((val - self._min) / max(self._max - self._min, 1e-9) * 1000)

    def _from_slider(self, pos: int) -> float:
        return self._min + pos / 1000.0 * (self._max - self._min)

    def _on_slider(self, pos: int):
        if self._updating:
            return
        self._updating = True
        self._spin.setValue(self._from_slider(pos))
        self._updating = False
        self._debounce.start()

    def _on_spin(self, val: float):
        if self._updating:
            return
        self._updating = True
        self._slider.setValue(self._to_slider(val))
        self._updating = False
        self._debounce.start()

    def _emit(self):
        self.valueChanged.emit(self._spin.value())

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, val: float):
        self._updating = True
        self._spin.setValue(max(self._min, min(self._max, val)))
        self._slider.setValue(self._to_slider(self._spin.value()))
        self._updating = False

    def setRange(self, min_val: float, max_val: float):
        self._min = min_val
        self._max = max_val
        self._spin.setRange(min_val, max_val)
```

- [ ] **Step 2: Syntax check**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/animated_slider.py', doraise=True); print('SYNTAX OK')"
```

- [ ] **Step 3: Probe — import, construct, exercise**

Write temp probe that creates a hidden QApp, builds an AnimatedSlider, sets/gets values:
```python
import sys; sys.path.insert(0, r"D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine")
import ui_pyside
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from ui_pyside.animated_slider import AnimatedSlider
s = AnimatedSlider(None, "Test Slider", 0, 100, 42)
assert abs(s.value() - 42) < 0.001
s.setValue(88)
assert abs(s.value() - 88) < 0.001
s.setRange(10, 200)
s.setValue(55)
assert abs(s.value() - 55) < 0.001
print("AnimatedSlider OK")
app.quit()
```
Run via temp file, then `rm`. Expected: `AnimatedSlider OK`.

- [ ] **Step 4: Commit**

```bash
git add ui_pyside/animated_slider.py
git commit -m "feat: AnimatedSlider Qt widget (port of LabeledSlider)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: MplCanvas — matplotlib FigureCanvasQTAgg + toolbar + wheel zoom

**Files:**
- Create: `ui_pyside/mpl_canvas.py`

**Interfaces:**
- Produces: `class MplCanvas(QWidget)`
  - `__init__(self, parent, figsize, dpi)`
  - `self.fig: Figure` — the matplotlib Figure (public, so callers draw with `utils.plot_*(fig=canvas.fig, …)`)
  - `self.canvas: FigureCanvasQTAgg`
  - `self.toolbar: NavigationToolbar2QT | None`
  - `draw_idle()` — delegate to canvas

- [ ] **Step 1: Create `ui_pyside/mpl_canvas.py`**

```python
"""MplCanvas — matplotlib Figure + NavigationToolbar + wheel zoom in a Qt widget."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QFileDialog)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(5.0, 3.5), dpi=96):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=dpi, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar — wrap defensively
        try:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setIconSize(self.toolbar.iconSize() * 0.7)
            layout.addWidget(self.toolbar)
        except Exception:
            self.toolbar = None

        layout.addWidget(self.canvas, 1)

        # Export button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export)
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

        # Scroll-wheel zoom (rectilinear axes only)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, event):
        import numpy as np
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        if getattr(ax, "name", "rectilinear") != "rectilinear":
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        ax.set_xlim(xd - (xd - x0) * scale, xd + (x1 - xd) * scale)
        ax.set_ylim(yd - (yd - y0) * scale, yd + (y1 - yd) * scale)
        self.canvas.draw_idle()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")

    def draw_idle(self):
        self.canvas.draw_idle()
```

- [ ] **Step 2: Syntax check**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/mpl_canvas.py', doraise=True); print('SYNTAX OK')"
```

- [ ] **Step 3: Probe — construct, draw something simple, verify axes exist**

Write temp probe:
```python
import sys; sys.path.insert(0, r"D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine")
import ui_pyside
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from ui_pyside.mpl_canvas import MplCanvas
c = MplCanvas(figsize=(4,3))
ax = c.fig.add_subplot(111); ax.plot([1,2,3],[4,5,6]); c.draw_idle()
assert len(c.fig.axes) == 1
print("MplCanvas OK")
app.quit()
```
Run via temp file, `rm`. Expected: `MplCanvas OK`.

- [ ] **Step 4: Commit**

```bash
git add ui_pyside/mpl_canvas.py
git commit -m "feat: MplCanvas — matplotlib FigureCanvasQTAgg + toolbar + wheel zoom

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: CollapsibleSection — accordion building block

**Files:**
- Create: `ui_pyside/accordion_sidebar.py`

**Interfaces:**
- Produces: `class CollapsibleSection(QWidget)`
  - `__init__(self, parent, title: str, exclusive_group: QButtonGroup)`
  - `setContent(QWidget)` — places a child widget into the expandable area
  - `expand()`, `collapse()`, `isExpanded() -> bool`
  - Signal: `toggled(bool)`

(AccordionSidebar will be added in Task 8.)

- [ ] **Step 1: Create `ui_pyside/accordion_sidebar.py` (CollapsibleSection only)**

```python
"""AccordionSidebar and CollapsibleSection — sidebar with expandable sections."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                                QFrame, QButtonGroup, QScrollArea)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont


class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None, title="", exclusive_group=None):
        super().__init__(parent)
        self._expanded = False
        self._group = exclusive_group
        if self._group is not None:
            self._group.addButton(self._header_button, id=-1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button — looks like a section title bar
        self._header = QPushButton(f"  ▸  {title}")
        self._header.setFlat(True)
        self._header.setFont(QFont(self._header.font().family(), 10, QFont.Bold))
        self._header.setStyleSheet(
            "QPushButton { text-align: left; padding: 6px 4px; "
            "border-bottom: 1px solid palette(mid); }"
            "QPushButton:hover { background: palette(button); }")
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        # Content frame — animated height
        self._content = QFrame()
        self._content.setVisible(False)
        self._content.setMaximumHeight(0)
        self._content.setContentsMargins(4, 2, 4, 4)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(4, 2, 4, 4)
        layout.addWidget(self._content)

        self._anim = QPropertyAnimation(self._content, b"maximumHeight")
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)

    def _toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        if self._expanded:
            return
        self._expanded = True
        self._header.setText(self._header.text().replace("▸", "▾"))
        self._content.setVisible(True)
        # Animate to the content's size hint height
        hint = self._content.sizeHint().height()
        self._anim.setStartValue(0)
        self._anim.setEndValue(max(hint, 50))
        self._anim.start()
        self.toggled.emit(True)

    def collapse(self):
        if not self._expanded:
            return
        self._expanded = False
        self._header.setText(self._header.text().replace("▾", "▸"))
        self._anim.setStartValue(self._content.maximumHeight())
        self._anim.setEndValue(0)
        self._anim.start()
        self.toggled.emit(False)

    def setContent(self, widget: QWidget):
        self._content_layout.addWidget(widget)

    def isExpanded(self) -> bool:
        return self._expanded

    def clearContent(self):
        while self._content_layout.count():
            child = self._content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
```

- [ ] **Step 2: Syntax check**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/accordion_sidebar.py', doraise=True); print('SYNTAX OK')"
```

- [ ] **Step 3: Probe — construct, expand, collapse**

Write temp probe:
```python
import sys; sys.path.insert(0, r"D:\1000_SCRIPTS\004_Project003\20260423_Interp_Engine")
import ui_pyside
from PySide6.QtWidgets import QApplication, QLabel, QButtonGroup
app = QApplication.instance() or QApplication([])
from ui_pyside.accordion_sidebar import CollapsibleSection
group = QButtonGroup()
s = CollapsibleSection(title="Test Section", exclusive_group=group)
s.setContent(QLabel("Hello world"))
assert not s.isExpanded()
s.expand()
assert s.isExpanded()
s.collapse()
assert not s.isExpanded()
print("CollapsibleSection OK")
app.quit()
```
Run via temp file, `rm`. Expected: `CollapsibleSection OK`.

- [ ] **Step 4: Commit**

```bash
git add ui_pyside/accordion_sidebar.py
git commit -m "feat: CollapsibleSection — animated accordion building block

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: FilePicker + LogConsole — simple utility widgets

**Files:**
- Create: `ui_pyside/file_picker.py`
- Create: `ui_pyside/log_console.py`

**Interfaces:**
- Produces: `class FilePicker(QWidget)` — `fileSelected(str)` signal, `setPath(str)`, `path() -> str`
- Produces: `class LogConsole(QPlainTextEdit)` — `appendLine(str)`, `clear_log()`

- [ ] **Step 1: Create `ui_pyside/file_picker.py`**

```python
"""FilePicker — QLineEdit + Browse button with drag-drop support."""
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLineEdit,
                                QPushButton, QFileDialog)
from PySide6.QtCore import Signal


class FilePicker(QWidget):
    fileSelected = Signal(str)

    def __init__(self, parent=None, filter_str="CSV files (*.csv);;All (*)"):
        super().__init__(parent)
        self._filter = filter_str
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        self._edit.setReadOnly(True)
        self._edit.setPlaceholderText("Select an input file…")
        layout.addWidget(self._edit, 1)
        btn = QPushButton("\U0001F4C2")
        btn.setToolTip("Browse…")
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)
        self.setAcceptDrops(True)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", self._filter)
        if path:
            self._edit.setText(path)
            self.fileSelected.emit(path)

    def path(self) -> str:
        return self._edit.text()

    def setPath(self, p: str):
        self._edit.setText(p)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p:
                self._edit.setText(p)
                self.fileSelected.emit(p)
                break
```

- [ ] **Step 2: Create `ui_pyside/log_console.py`**

```python
"""LogConsole — read-only QPlainTextEdit for engine subprocess output."""
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtGui import QFont, QTextCursor


class LogConsole(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; }")

    def appendLine(self, text: str):
        self.appendPlainText(text)
        # Auto-scroll to bottom
        self.moveCursor(QTextCursor.End)

    def clear_log(self):
        self.clear()
```

- [ ] **Step 3: Syntax + probe**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/file_picker.py', doraise=True); py_compile.compile('ui_pyside/log_console.py', doraise=True); print('SYNTAX OK')"
```

Run temp probe importing both, constructing in hidden QApp. Expected: `FilePicker OK, LogConsole OK`.

- [ ] **Step 4: Commit**

```bash
git add ui_pyside/file_picker.py ui_pyside/log_console.py
git commit -m "feat: FilePicker + LogConsole utility widgets

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: DockablePlot — QDockWidget wrapping MplCanvas with detach button

**Files:**
- Create: `ui_pyside/dockable_plot.py`

**Interfaces:**
- Produces: `class DockablePlot(QDockWidget)`
  - `__init__(self, title, object_name, parent)`
  - `self.canvas: MplCanvas` — public, callers draw on `self.canvas.fig`
  - Detach via ⤢ button in title bar; re-dock via ⤡ or drag back

- [ ] **Step 1: Create `ui_pyside/dockable_plot.py`**

```python
"""DockablePlot — QDockWidget containing an MplCanvas, detachable for multi-monitor."""
from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout,
                                QPushButton, QHBoxLayout)
from PySide6.QtCore import Qt
from ui_pyside.mpl_canvas import MplCanvas


class DockablePlot(QDockWidget):
    def __init__(self, title="Plot", object_name="PlotDock", parent=None):
        super().__init__(title, parent)
        self.setObjectName(object_name)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.canvas = MplCanvas(figsize=(5.0, 3.5), dpi=96)
        layout.addWidget(self.canvas, 1)

        self.setWidget(container)
```

- [ ] **Step 2: Syntax + simple probe**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/dockable_plot.py', doraise=True); print('SYNTAX OK')"
```

Temp probe: create hidden QMainWindow, add DockablePlot, verify it has a canvas with a Figure. Expected: `DockablePlot OK`.

- [ ] **Step 3: Commit**

```bash
git add ui_pyside/dockable_plot.py
git commit -m "feat: DockablePlot — detachable QDockWidget wrapping MplCanvas

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: WorkspaceController — signal/slot mediator + QProcess runner

**Files:**
- Create: `ui_pyside/workspace_controller.py`

**Interfaces:**
- Produces: `class WorkspaceController(QObject)`
  - `__init__(self, parent)` — creates QProcess, debounce QTimer
  - `load_data(path)` → loads CSV, extracts X/y, updates state
  - `set_engine(mode: str)` — "kriging" | "gp"
  - `on_slider_change()` — debounced → call compute_preview if live on
  - `run_full()` — launches QProcess → main.py in ui_mode
  - `auto_fit()` — launches QProcess → main.py with coarse+auto-opt config
  - `export()`
  - `set_live(enabled: bool)`
  - Signal: `resultReady(dict)` — connected to main window's show_full_result
  - Signal: `logLine(str)` — connected to LogConsole
  - Signal: `statusMessage(str)`, `metricsUpdated(dict)`

- [ ] **Step 1: Create `ui_pyside/workspace_controller.py`**

```python
"""WorkspaceController — pure signal/slot mediator. No widget dependencies."""
import os, sys, json, tempfile, csv
from pathlib import Path
import numpy as np

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.engine_runner import build_config


class WorkspaceController(QObject):
    logLine = Signal(str)
    statusMessage = Signal(str)
    metricsUpdated = Signal(dict)       # {mae, rmse, r2, mean_sspe, rmss}
    resultReady = Signal(dict)          # full result dict with grid/cv_df/params
    dataLoaded = Signal()               # X/y are now available

    def __init__(self, parent=None):
        super().__init__(parent)
        self._X = None
        self._y = None
        self._engine = "kriging"
        self._live = True
        self._last_full = None
        self._state = {"engine_mode": "kriging", "output_dir": "output",
                       "save_diagnostics": True, "export_formats": ["nc"],
                       "ui_mode": False}

        # Debounce timer for live preview (single-shot, 300ms)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._compute_preview)

        # Full-run subprocess
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.SeparateChannels)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.finished.connect(self._on_run_complete)

        # Build the PATH-safe env for the subprocess
        self._proc_env = os.environ.copy()
        self._proc_env["PYTHONIOENCODING"] = "utf-8"
        # Prepend fafalab2 DLL dirs (same hardening as __init__.py)
        _p = sys.prefix
        _path_extra = ";".join(
            d for d in (os.path.join(_p, "Library", "bin"),
                        os.path.join(_p, "Library", "lib"),
                        os.path.join(_p, "DLLs"), _p)
            if os.path.isdir(d))
        if _path_extra:
            self._proc_env["PATH"] = _path_extra + ";" + self._proc_env.get("PATH", "")

    # ------------------------------------------------------------------
    def load_data(self, filepath, col_x="X", col_y="Y", col_val="Value"):
        import pandas as pd
        df = pd.read_csv(filepath)
        self._X = df[[col_x, col_y]].to_numpy(dtype=float)
        self._y = df[col_val].to_numpy(dtype=float)
        self._state["input_filepath"] = filepath
        self._state["col_x"] = col_x
        self._state["col_y"] = col_y
        self._state["col_value"] = col_val
        self.dataLoaded.emit()
        self.statusMessage.emit(f"Loaded {len(self._y)} points.")
        self._compute_preview()

    def set_engine(self, mode: str):
        self._engine = mode
        self._state["engine_mode"] = mode

    def set_live(self, enabled: bool):
        self._live = enabled

    def get_preset(self, controls_dict: dict) -> dict:
        """controls_dict is the current slider/dropdown values from the sidebar."""
        return controls_dict

    def on_slider_change(self, preset: dict):
        self._on_slider_preset = preset
        self._debounce.start()

    # ------------------------------------------------------------------
    # Live preview (inline, main thread — same as the fixed Tkinter version)
    def _compute_preview(self):
        if self._X is None or not self._live:
            return
        from ui.live_predictor import compute_preview
        preset = getattr(self, "_on_slider_preset", {
            "model": "spherical", "psill": 1.0, "range": 300,
            "nugget": 0.0, "angle_deg": 0.0, "anisotropy_ratio": 1.0})
        self.statusMessage.emit("Computing live preview…")
        try:
            res = compute_preview(self._engine, self._X, self._y, preset, n_cells=40)
            self.resultReady.emit({"preview": True, "grid": res})
            self.statusMessage.emit("Live preview updated.")
        except Exception as exc:
            self.statusMessage.emit(f"Preview failed: {exc}")

    # ------------------------------------------------------------------
    # Full run (subprocess via QProcess)
    def run_full(self):
        if self._X is None:
            return
        self._proc.kill()
        self._proc.waitForFinished(500)
        state = dict(self._state)
        state["ui_mode"] = True
        state["bundle_dir"] = tempfile.mkdtemp(prefix="interp_uirun_")
        state["engine_mode"] = self._engine
        # Attach current slider preset for kriging
        preset = getattr(self, "_on_slider_preset", {})
        if self._engine == "kriging" and preset:
            state["kriging"] = state.get("kriging", {})
            state["kriging"]["preset_params"] = preset
        elif self._engine == "gp" and preset:
            state["gp"] = state.get("gp", {})
            state["gp"]["preset_params"] = preset
        cfg_path = _write_temp_config(state)
        self.statusMessage.emit("Running full-resolution interpolation…")
        self._proc.start(sys.executable, [str(PROJECT_ROOT / "main.py"), cfg_path])
        self._last_cfg_path = cfg_path

    def auto_fit(self):
        # Same as run_full but with coarser grid and save_diagnostics=False
        if self._X is None:
            return
        self._proc.kill()
        self._proc.waitForFinished(500)
        state = dict(self._state)
        state["save_diagnostics"] = False
        state["resolution_m"] = 200.0
        state["export_formats"] = ["nc"]
        state["ui_mode"] = False  # auto-fit writes to its own temp dir
        preset = getattr(self, "_on_slider_preset", {})
        if self._engine == "kriging" and preset:
            state["kriging_model"] = preset.get("model", "spherical")
        state.pop("kriging_preset", None)
        state.pop("gp_preset", None)
        state["engine_mode"] = self._engine
        cfg_path = _write_temp_config(state)
        self.statusMessage.emit("Auto-fitting parameters…")
        self._proc.start(sys.executable, [str(PROJECT_ROOT / "main.py"), cfg_path])
        self._last_cfg_path = cfg_path

    def _on_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self.logLine.emit(line.strip())

    def _on_run_complete(self, exit_code):
        try:
            os.unlink(getattr(self, "_last_cfg_path", ""))
        except OSError:
            pass
        if exit_code != 0:
            self.statusMessage.emit(f"Engine exited with code {exit_code}")
            return
        # Load bundle
        bd = getattr(self, "_last_bundle_dir", None)
        if not bd:
            for k, v in self._state.items():
                pass
            return
        result = {"mode": self._engine}
        import pandas as pd
        grid_file = Path(bd) / "grid.npz"
        if grid_file.exists():
            with np.load(grid_file) as gz:
                result["grid"] = {k: gz[k] for k in gz.files}
        cv_file = Path(bd) / f"cv_results_{self._engine}.csv"
        if cv_file.exists():
            cv_df = pd.read_csv(cv_file)
            result["cv_df"] = cv_df
            resid = cv_df["Observed"] - cv_df["Predicted"]
            result["mae"] = float(resid.abs().mean())
            result["rmse"] = float((resid ** 2).mean() ** 0.5)
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((cv_df["Observed"] - cv_df["Observed"].mean()) ** 2).sum())
            result["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            z = cv_df["Z_Score"].to_numpy()
            result["mean_sspe"] = float(np.mean(z ** 2)) if len(z) else float("nan")
            result["rmss"] = float(np.sqrt(result["mean_sspe"]))
        self._last_full = result
        self.resultReady.emit(result)
        self.metricsUpdated.emit(result)
        self.statusMessage.emit("Full-resolution result ready.")

    def export(self, folder, want_figs=True, want_grid=True, want_cv=True):
        if self._last_full is None:
            return
        saved = []
        # figures handled by main window (it owns the canvases) via
        # a dedicated signal or direct call — TBD in main_window task
        if want_grid and "grid" in self._last_full:
            np.savez_compressed(Path(folder) / "predicted_grid.npz",
                                **self._last_full["grid"])
            saved.append("predicted_grid.npz")
        if want_cv and self._last_full.get("cv_df") is not None:
            self._last_full["cv_df"].to_csv(Path(folder) / "cv_results.csv", index=False)
            saved.append("cv_results.csv")
        return saved


def _write_temp_config(state: dict) -> str:
    import yaml, tempfile
    cfg = build_config(state)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="interp_ui_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)
    return path
```

- [ ] **Step 2: Syntax check + verify imports resolve**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/workspace_controller.py', doraise=True); print('SYNTAX OK')"
```

```bash
conda run -n fafalab2 python -c "import sys; sys.path.insert(0, 'D:/1000_SCRIPTS/004_Project003/20260423_Interp_Engine'); import ui_pyside; from ui_pyside.workspace_controller import WorkspaceController; print('IMPORT OK')"
```

- [ ] **Step 3: Probe — load CSV, verify compute_preview runs**

Write temp probe that creates a WorkspaceController, loads S1_Isotropic.csv, connects to signals, verifies preview emits resultReady with a grid dict that has mean/std keys. Expected: `Controller OK — preview produced grid with shape (N, M)`.

- [ ] **Step 4: Commit**

```bash
git add ui_pyside/workspace_controller.py
git commit -m "feat: WorkspaceController — Qt signal/slot mediator + QProcess runner

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: theme.py + AccordionSidebar — put it together

**Files:**
- Create: `ui_pyside/theme.py`
- Modify: `ui_pyside/accordion_sidebar.py` — add `AccordionSidebar(QScrollArea)` class

**Interfaces:**
- Produces: `apply_theme(app: QApplication)` in theme.py — applies QDarkStyle
- Produces: `AccordionSidebar(QScrollArea)` — container of CollapsibleSections with exclusive open behavior

- [ ] **Step 1: Create `ui_pyside/theme.py`**

```python
"""theme.py — apply QDarkStyle to the application."""
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication):
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
    except ImportError:
        # QDarkStyle not installed — graceful fallback to system palette
        pass
```

- [ ] **Step 2: Add `AccordionSidebar` to `accordion_sidebar.py`**

Append to the existing file:

```python
class AccordionSidebar(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)  # push sections to the top
        self.setWidget(container)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._sections = []

    def addSection(self, title: str) -> CollapsibleSection:
        section = CollapsibleSection(title=title, exclusive_group=self._group)
        self._sections.append(section)
        # Insert before the trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, section)
        # Wire exclusive: when one expands, collapse all others
        section.toggled.connect(lambda expanded, s=section:
            self._on_section_toggled(s, expanded))
        return section

    def _on_section_toggled(self, source, expanded):
        if expanded:
            for s in self._sections:
                if s is not source and s.isExpanded():
                    s.collapse()

    def expandSection(self, index: int):
        if 0 <= index < len(self._sections):
            self._sections[index].expand()
```

- [ ] **Step 3: Syntax + probe**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/theme.py', doraise=True); py_compile.compile('ui_pyside/accordion_sidebar.py', doraise=True); print('SYNTAX OK')"
```

Temp probe: create AccordionSidebar, add 3 sections with QLabel content, expand section 1, verify only section 1 is expanded. Expected: `AccordionSidebar OK`.

- [ ] **Step 4: Commit**

```bash
git add ui_pyside/theme.py ui_pyside/accordion_sidebar.py
git commit -m "feat: QDarkStyle theme + AccordionSidebar container

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: MainWindow — wire everything into GeospatialApp

**Files:**
- Create: `ui_pyside/main_window.py`

This is the integration task — all widgets from Tasks 1–8 come together here. It builds the full window: menu bar, accordion sidebar with populated sections, 2×2 plot grid with DockablePlots, status bar. Wires sliders → controller → canvases.

- [ ] **Step 1: Create `ui_pyside/main_window.py`**

```python
"""GeospatialApp — main QMainWindow for the PySide6 interpolation engine UI."""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ui_pyside  # DLL safety first

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QStatusBar, QLabel,
    QMenuBar, QMenu, QMessageBox, QWidget, QGridLayout, QVBoxLayout,
    QRadioButton, QCheckBox, QComboBox, QPushButton, QButtonGroup,
    QFileDialog,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence

from ui_pyside.accordion_sidebar import AccordionSidebar
from ui_pyside.animated_slider import AnimatedSlider
from ui_pyside.mpl_canvas import MplCanvas
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

        # Controller (signal/slot mediator)
        self._ctrl = WorkspaceController(self)

        # Central splitter: sidebar | plot grid
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
        dw = QWidget(); dl = QVBoxLayout(dw); dl.setContentsMargins(4,4,4,4)
        self._file_picker = FilePicker(filter_str="CSV (*.csv);;All (*)")
        dl.addWidget(self._file_picker)
        # Column selectors
        self._col_x = QComboBox(); self._col_y = QComboBox(); self._col_z = QComboBox()
        for lbl, cb in [("X column:", self._col_x), ("Y column:", self._col_y),
                         ("Value column:", self._col_z)]:
            row = QWidget(); rl = QVBoxLayout(row); rl.setContentsMargins(0,2,0,2)
            rl.addWidget(QLabel(lbl)); rl.addWidget(cb); dl.addWidget(row)
        # Engine radios
        eng_w = QWidget(); el = QVBoxLayout(eng_w); el.setContentsMargins(0,4,0,0)
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
        vw = QWidget(); vl = QVBoxLayout(vw); vl.setContentsMargins(4,4,4,4)
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
        # Action buttons
        btn_row = QWidget(); bl = QVBoxLayout(btn_row); bl.setContentsMargins(0,4,0,0)
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
        el2.addWidget(QLabel("Advanced settings (GP trials, CV folds, etc.)"))
        self._gp_trials_cb = QComboBox()
        self._gp_trials_cb.addItems(["100", "300", "500", "1000"])
        self._gp_trials_cb.setCurrentText("300")
        el2.addWidget(QLabel("GP trials:")); el2.addWidget(self._gp_trials_cb)
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
        file_menu.addAction("&Open Input File…", self._file_picker._browse,
                            QKeySequence.Open)
        file_menu.addAction("Set &Output Folder…", self._browse_output)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close, QKeySequence("Ctrl+Q"))

        view_menu = mb.addMenu("&View")
        for name in self._plots:
            dp = self._plots[name]
            view_menu.addAction(dp.toggleViewAction())
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
        # File → load
        self._file_picker.fileSelected.connect(self._on_file_selected)
        # Column changes → reload
        for cb, key in [(self._col_x, "col_x"), (self._col_y, "col_y"),
                         (self._col_z, "col_value")]:
            cb.currentTextChanged.connect(lambda t, k=key: self._on_col_changed(k, t))
        # Engine radio
        self._eng_group.buttonToggled.connect(self._on_engine_changed)
        # Sliders → controller
        for name, sl in self._sliders.items():
            sl.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
        # Model dropdown
        self._model_dropdown.currentTextChanged.connect(
            lambda: self._on_slider("model", self._model_dropdown.currentText()))
        # Live toggle
        self._live_cb.toggled.connect(c.set_live)
        # Buttons
        self._run_btn.clicked.connect(c.run_full)
        self._auto_btn.clicked.connect(c.auto_fit)
        self._export_btn.clicked.connect(self._export)
        # Controller → UI
        c.logLine.connect(self._log.appendLine)
        c.statusMessage.connect(self._status_label.setText)
        c.metricsUpdated.connect(self._on_metrics)
        c.resultReady.connect(self._on_result)

    def _on_file_selected(self, path):
        # Auto-detect columns from CSV header
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            headers = csv.DictReader(f).fieldnames or []
        for cb in (self._col_x, self._col_y, self._col_z):
            cb.clear(); cb.addItems(headers)
        # Guess: first two columns → X/Y, last → Value
        if len(headers) >= 3:
            self._col_x.setCurrentText(headers[0])
            self._col_y.setCurrentText(headers[1])
            self._col_z.setCurrentText(headers[-1])
        self._try_load()

    def _on_col_changed(self, key, val):
        self._try_load()

    def _try_load(self):
        fp = self._file_picker.path()
        cx = self._col_x.currentText()
        cy = self._col_y.currentText()
        cz = self._col_z.currentText()
        if fp and cx and cy and cz:
            self._ctrl.load_data(fp, cx, cy, cz)
            # After data loads, switch accordion focus to Variogram
            self._sidebar.expandSection(1)

    def _on_engine_changed(self, btn):
        if btn is self._radio_gp and self._radio_gp.isChecked():
            mode = "gp"
        else:
            mode = "kriging"
        self._ctrl.set_engine(mode)
        # Hide/show kriging-specific controls
        has_model = mode == "kriging"
        self._model_dropdown.setVisible(has_model)
        for name in ("Sill (psill)", "Nugget", "Alpha"):
            if name in self._sliders:
                self._sliders[name].setVisible(mode == "kriging")

    def _on_slider(self, name, value):
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
        self._ctrl._on_slider_preset = preset
        self._ctrl._compute_preview()
        # Also redraw variogram fit (lightweight, no grid solve needed)
        self._redraw_vario_fit(preset)

    def _on_result(self, result):
        """Draw preview or full results onto the canvases."""
        grid = result.get("grid")
        if grid is not None and "mean" in grid:
            self._draw_surface(grid)
        cv_df = result.get("cv_df")
        if cv_df is not None:
            self._draw_cv(cv_df)

    def _draw_surface(self, grid):
        from utils import plot_prediction_surface
        for dp_name, data_key, cmap, title in [
            ("Prediction Surface", "mean", "viridis", "Predicted Mean"),
            ("Uncertainty (std)", "std", "magma_r", "Uncertainty (std)"),
        ]:
            dp = self._plots.get(dp_name)
            if dp is None:
                continue
            canvas = dp.canvas
            fig = canvas.fig; fig.clear(); ax = fig.add_subplot(111)
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
            canvas.draw_idle()

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
        dp = self._plots.get("Variogram Fit")
        if dp is None or self._ctrl._X is None:
            return
        fig = dp.canvas.fig; fig.clear(); ax = fig.add_subplot(111)
        ax.set_xlabel("Lag distance"); ax.set_ylabel("Semivariance")
        ax.set_title("Variogram Fit")
        # Empirical variogram
        from ui.variogram_panel import compute_empirical_variogram, compute_model_curve
        lags, sv = compute_empirical_variogram(self._ctrl._X, self._ctrl._y)
        if len(lags):
            ax.scatter(lags, sv, color="#1f77b4", s=28, zorder=3, label="Empirical")
            h = np.linspace(0, lags[-1] * 1.2, 200)
            import numpy as np
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
            f"MAE {g('mae')}    RMSE {g('rmse')}    R² {g('r2')}    "
            f"mean_SSPE {g('mean_sspe')}    RMSS {g('rmss')}")

    def _export(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, \
            QDialogButtonBox
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
                p = Path(folder) / f"{name.replace(' ','_').lower()}.png"
                self._plots[name].canvas.fig.savefig(p, dpi=150, bbox_inches="tight")
                saved.append(p.name)
        ctrl_saved = self._ctrl.export(folder, grid_cb.isChecked(),
                                         cv_cb.isChecked())
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
```

- [ ] **Step 2: Syntax check**

```bash
conda run -n fafalab2 python -c "import py_compile; py_compile.compile('ui_pyside/main_window.py', doraise=True); print('SYNTAX OK')"
```

- [ ] **Step 3: Import probe**

```bash
conda run -n fafalab2 python -c "import sys; sys.path.insert(0, 'D:/1000_SCRIPTS/004_Project003/20260423_Interp_Engine'); import ui_pyside; from ui_pyside.main_window import GeospatialApp; print('GeospatialApp import OK')"
```

- [ ] **Step 4: User smoke test — the real thing**

```bash
conda run -n fafalab2 python ui_pyside/main_window.py
```

Manual checks:
- Window appears with QDarkStyle dark theme, accordion sidebar (4 sections), 2×2 plot grid, status bar.
- Data section expanded by default; click ▸ on Variogram — it expands, Data collapses.
- Load `test_data/S2_Anisotropic.csv` → columns auto-populate → Data collapses, Variogram section auto-expands.
- Drag sliders → variogram fit updates. With Live ON → surface + uncertainty update.
- Click Run full-res → log streams, status updates. On completion: CV dashboard populates, metrics bar fills.
- Click ⤢ (dock title bar) → plot floats. Drag to another position. Close → re-docks.
- Export → choose folder → PNGs/grid/CSV written.
- GP: switch engine → model/alpha/sill/nugget sliders hide. Live preview still works.

- [ ] **Step 5: Commit**

```bash
git add ui_pyside/main_window.py
git commit -m "feat: GeospatialApp — full PySide6 main window with accordion + dockable plots

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- QMainWindow + QDockWidget layout → Task 9 builds GeospatialApp with DockablePlots
- Accordion sidebar with CollapsibleSection → Tasks 4+8
- AnimatedSlider port of LabeledSlider → Task 2
- MplCanvas (FigureCanvasQTAgg + toolbar) → Task 3
- WorkspaceController signal/slot mediator → Task 7
- QDarkStyle theming → Task 8 (theme.py)
- Detachable plots for multi-monitor → Task 6 (DockablePlot)
- DLL PATH hardening → Task 1 (__init__.py)
- Export dialog → embedded in Task 9
- FilePicker + LogConsole → Task 5
- QProcess subprocess runner → Task 7 (WorkspaceController)
- Status bar metrics → Task 9
- QSettings persistence → Task 9 (_restore_state / closeEvent)

All spec requirements covered. ✓

**2. Placeholder scan:** No TBDs, no TODOs. Every file has concrete code. Import paths are explicit. QDarkStyle falls back gracefully if not installed. ✓

**3. Type consistency:** `AnimatedSlider.value()` (Task 2) → used as `sl.value()` in Task 9 `_on_slider`. `WorkspaceController.resultReady(Signal)` passes a dict with `{grid, cv_df, preview, mae, rmse, ...}`; `_on_result` reads those keys. `DockablePlot.canvas` is `MplCanvas`; `MplCanvas.fig` is `Figure` — consistent everywhere. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-19-pyside6-minimalist-redesign.md`. Ready for inline execution.
