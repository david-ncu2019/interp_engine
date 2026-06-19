"""theme.py — apply QDarkStyle to the application."""
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication):
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
    except ImportError:
        pass  # QDarkStyle not installed — graceful fallback
