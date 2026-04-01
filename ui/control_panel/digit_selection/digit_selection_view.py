from PyQt5.QtWidgets import QWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt



class DigitWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels = []
        self.boxes  = []
        self.box_all = QCheckBox()
        layout = QVBoxLayout()
        layout_top    = QHBoxLayout()
        layout_bottom = QHBoxLayout()
        label_top = QLabel('D I G I T S   T O   L E A R N', alignment = Qt.AlignCenter)
        label_top_2 = QLabel('select all', alignment = Qt.AlignCenter)
        self.box_all.setChecked(True)
        self.box_all.stateChanged.connect(self.choose_all_digits)
        layout_top.addStretch()
        layout_top.addWidget(label_top)
        layout_top.addWidget(label_top_2)
        layout_top.addWidget(self.box_all)
        layout_top.addStretch()
        for digit in range(10):
            layout_d = QVBoxLayout()
            label = QLabel(str(digit), alignment = Qt.AlignCenter)
            box   = QCheckBox()
            box.setChecked(True)
            self.labels.append(label)
            self.boxes .append(box)
            layout_d.addWidget(box, alignment = Qt.AlignCenter)
            layout_d.addWidget(label, alignment = Qt.AlignCenter)
            layout_bottom.addLayout(layout_d)
        layout.addLayout(layout_top)
        layout.addLayout(layout_bottom)
        layout.addStretch()
        self.setLayout(layout)

    def choose_all_digits(self):
        for box in self.boxes:
            box.setChecked(self.box_all.isChecked())

    def get_checked(self):
        data = self.parent().parent().parent().model.data
        checked =  [ i for i in range(len(self.boxes)) if self.boxes[i].isChecked() ]
        return checked
