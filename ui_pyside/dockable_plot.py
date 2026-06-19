"""DockablePlot — QDockWidget containing an MplCanvas, detachable for multi-monitor."""
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
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
