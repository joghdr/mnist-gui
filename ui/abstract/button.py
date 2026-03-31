from PyQt5.QtWidgets import QPushButton


class Button(QPushButton):
    def __init__(self, label, run, parent=None, enabled=True):
        super().__init__(parent)
        self.setText(label)
        self.setFixedHeight(45)
        self.setEnabled(enabled)
        self.clicked.connect(run)
