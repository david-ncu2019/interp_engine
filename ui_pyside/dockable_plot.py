"""PlotPanel — QWidget containing an MplCanvas, for the 2×2 plot grid."""
from PySide6.QtWidgets import QWidget, QVBoxLayout
from ui_pyside.mpl_canvas import MplCanvas


class PlotPanel(QWidget):
    def __init__(self, title="Plot", parent=None):
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.canvas = MplCanvas(figsize=(5.0, 3.5), dpi=96)
        layout.addWidget(self.canvas, 1)

    def title(self) -> str:
        return self._title
