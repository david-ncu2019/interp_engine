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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button — section title bar
        self._header = QPushButton(f"  ▸  {title}")
        self._header.setFlat(True)
        self._header.setFont(QFont(self._header.font().family(), 10, QFont.Bold))
        self._header.setStyleSheet(
            "QPushButton { text-align: left; padding: 6px 4px; "
            "border-bottom: 1px solid palette(mid); }"
            "QPushButton:hover { background: palette(button); }")
        self._header.clicked.connect(self._toggle)

        if self._group is not None:
            self._group.addButton(self._header)

        layout.addWidget(self._header)

        # Content frame — animated height
        self._content = QFrame()
        self._content.setVisible(False)
        self._content.setMaximumHeight(0)
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
        self._layout.addStretch(1)
        self.setWidget(container)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._sections = []

    def addSection(self, title: str) -> CollapsibleSection:
        section = CollapsibleSection(title=title, exclusive_group=self._group)
        self._sections.append(section)
        self._layout.insertWidget(self._layout.count() - 1, section)
        section.toggled.connect(
            lambda expanded, s=section: self._on_section_toggled(s, expanded))
        return section

    def _on_section_toggled(self, source, expanded):
        if expanded:
            for s in self._sections:
                if s is not source and s.isExpanded():
                    s.collapse()

    def expandSection(self, index: int):
        if 0 <= index < len(self._sections):
            self._sections[index].expand()
