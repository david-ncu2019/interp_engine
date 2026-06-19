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
