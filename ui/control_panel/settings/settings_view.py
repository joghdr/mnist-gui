from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QDoubleSpinBox
from PyQt5.QtCore import Qt

from ui.control_panel.layer_architecture.layer_architecture_view import HiddenLayersWidget
from ui.abstract.integer_choice import IntegerChoice, IntegerChoiceWidget
from core.functions import activation_dict

class FloatChoiceWidget(QWidget):

    def __init__(self, label, Range, decimals = 2, default_value = 1, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label  = QLabel(label, alignment = Qt.AlignCenter)
        self.choices = QDoubleSpinBox()
        self.choices.setAlignment(Qt.AlignRight)
        self.choices.setRange(Range[0], Range[1])
        self.choices.setSingleStep(Range[2])
        self.choices.setDecimals(decimals)
        self.choices.setValue(default_value)
        layout.addWidget(label)
        layout.addWidget(self.choices)

        self.setLayout(layout)

class ConfigWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = self.parent().parent()
        self.config = self.main_window.config
        self.layout = QGridLayout()
        self.hidden_nodes = HiddenLayersWidget(parent=self)
        self.activation = IntegerChoiceWidget('activation', activation_dict.keys(),
                                              self.config.activation_hidden, parent=self)
        self.epochs = IntegerChoice('epochs', [0, 1000], self.config.epochs, parent=self)
        self.sample = IntegerChoice('sample (train)', [1, 42000], self.config.sample, parent=self)
        self.batch_size = IntegerChoice('batch size', [25,42000], self.config.batch_size, parent=self)

        self.learning_rate = FloatChoiceWidget('learning rate', [0.0, 10, 1.0], 6,
                                               self.config.learning_rate, parent=self)

        self.ratio = FloatChoiceWidget('pts train / pts total ', [0.1, 0.99, 0.1], 2,
                                              self.config.ratio, parent=self)


        self.layout.addWidget(self.hidden_nodes,  0, 0)
        self.layout.addWidget(self.sample,        1, 0)
        self.layout.addWidget(self.activation,    2, 0)

        self.layout.addWidget(self.epochs,        0, 1)
        self.layout.addWidget(self.ratio,         1, 1)

        self.layout.addWidget(self.learning_rate, 0, 2)
        self.layout.addWidget(self.batch_size   , 1, 2)

        self.setLayout(self.layout)

    def update_config_from_panel(self, force_weights=False):
        self.config.input_dimension = 28*28
        self.config.output_dimension = 10
        self.config.hidden_nodes = self.hidden_nodes.hidden_nodes
        self.config.sample = self.sample.choices.value()
        self.config.activation_hidden = self.activation.choices.currentText()
        self.config.epochs = self.epochs.choices.value()
        self.config.ratio = self.ratio.choices.value()
        self.config.learning_rate = self.learning_rate.choices.value()
        self.config.batch_size = self.batch_size.choices.value()
        self.config.digits = self.main_window.control_panel.digit_panel.get_checked()

        self.main_window.model.sync_to(self.config, force_weights=force_weights)
        self.main_window.model.data.config_settings(self.config)

        self.main_window.accuracy_panel.init_axes(self.config.epochs)
        self.main_window.cost_panel.init_axes(self.config.epochs)
