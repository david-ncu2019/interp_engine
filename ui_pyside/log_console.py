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
        self.moveCursor(QTextCursor.End)

    def clear_log(self):
        self.clear()
