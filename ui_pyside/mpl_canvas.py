"""MplCanvas — matplotlib Figure + NavigationToolbar + wheel zoom in Qt."""
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

        # Export button row
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
