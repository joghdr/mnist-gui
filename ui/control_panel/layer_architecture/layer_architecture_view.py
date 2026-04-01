from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox
from PyQt5.QtCore import Qt

from ui.abstract.integer_choice import IntegerChoice, IntegerChoiceWidget
from ui.abstract.base_plot import PLOT_COST_HEIGHT, PLOT_COST_WIDTH




class LayerConfigWidget(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.Window)      #allows window to open (needed since parent is passed)
        self.setFixedSize(PLOT_COST_WIDTH // 2, PLOT_COST_HEIGHT // 3)
        self.hidden_nodes = self.parent().hidden_nodes
        self.n_of_layers = len(self.hidden_nodes)
        self.layers = []
        layout0 = QVBoxLayout()
        self.setWindowTitle('Layer Configuration')
        self.label = QLabel('Layer Configuration', alignment = Qt.AlignCenter)
        font = QWidget().font()
        for layer in range(self.n_of_layers):
            layout = QVBoxLayout()
            label = QLabel(f'hidden layer {layer + 1}', alignment = Qt.AlignCenter)
            values = QSpinBox()
            values.setRange(1,40)
            values.setValue(self.hidden_nodes[layer])
            values.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            layout.addWidget(values)
            layout0.addLayout(layout)
            self.layers.append((label, values))
        for i in range(len(self.layers)):
            self.layers[i][1].valueChanged.connect(self.set_hidden_nodes)
            print(self.layers[i][1].value())
        self.setLayout(layout0)

    def set_hidden_nodes(self, i):
        for i in range(len(self.layers)):
            self.hidden_nodes[i] = self.layers[i][1].value()
        print('-----------------------------')
        for i in range(len(self.hidden_nodes)):
            print(f'there are {self.hidden_nodes[i]} nodes in layer {i}')

class HiddenLayersWidget(IntegerChoiceWidget):

    def __init__(self, parent):
        hidden_nodes = parent.parent().parent().model.config.hidden_nodes
        super().__init__('hidden layers', [str(i) for i in range(4)], default=str(len(hidden_nodes)), parent=parent)
        if hidden_nodes is None:
            self.hidden_nodes = []
        else:
            self.hidden_nodes = hidden_nodes
        self.window = None
        self.choices.currentTextChanged.connect(self.update_hidden_nodes)

    def set_hidden_nodes(self):
        self.hidden_nodes = [1 for i in range(self.hidden_nodes)]

    def update_hidden_nodes(self):
        self.hidden_nodes = int(self.choices.currentText())
        self.set_hidden_nodes()
        if len(self.hidden_nodes) != 0:
            self.window = LayerConfigWidget(parent=self)
            self.window.show()
            print(f'self.hidden_nodes != 0')
        else:
            print(f'self.hidden_nodes = 0')
            if self.window is not None and self.window.isVisible():
                self.window.close()
