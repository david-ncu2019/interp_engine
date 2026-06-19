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
