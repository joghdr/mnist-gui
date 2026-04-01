from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QSpinBox
from PyQt5.QtCore import Qt

class IntegerChoice(QWidget):

    def __init__(self, label, Range, default_value, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label  = QLabel(label, alignment = Qt.AlignCenter)
        self.choices = QSpinBox()
        self.choices.setRange(Range[0], Range[1])
        self.choices.setValue(default_value)
        self.choices.setAlignment(Qt.AlignRight)
        layout.addWidget(label)
        layout.addWidget(self.choices)

        self.setLayout(layout)


class IntegerChoiceWidget(QWidget):

    def __init__(self, label, items, default, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel(label, alignment = Qt.AlignCenter)
        self.choices = QComboBox()
        self.choices.addItems(items)
        self.choices.setLayoutDirection(Qt.LeftToRight)
        default_index = list(items).index(default)
        self.choices.setCurrentIndex(default_index)
        layout.addWidget(label)
        layout.addWidget(self.choices)
        self.setLayout(layout)
