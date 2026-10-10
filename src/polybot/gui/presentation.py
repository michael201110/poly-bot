"""Shared presentation widgets for the desktop workspace."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QToolButton, QVBoxLayout, QWidget


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Display readable numbers while retaining the editor's full precision."""

    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        separator = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(separator) if separator in text else text


class ActionGrid(QGridLayout):
    """Keep action labels readable instead of squeezing them into a single row."""

    def __init__(self, columns: int = 3) -> None:
        super().__init__()
        self.columns = columns
        self.setSpacing(8)

    def addWidget(self, widget: QWidget, *args) -> None:
        if args:
            super().addWidget(widget, *args)
        else:
            index = self.count()
            super().addWidget(widget, index // self.columns, index % self.columns)


class ExpandableSection(QWidget):
    """A named workflow whose controls remain available without filling the page."""

    def __init__(self, title: str, content: QWidget) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setObjectName("sectionToggle")
        self.toggle.setToolTip(f"Expand {title.lower()}")
        self.content = content
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        content.hide()
        self.toggle.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(expanded)

    def expand(self) -> None:
        self.toggle.setChecked(True)
