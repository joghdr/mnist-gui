from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QHBoxLayout, QVBoxLayout, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt

from core.Config import Config
from core.nn import nn
from ui.diagnostics_panel.predict_test_set.predict_test_set_view import ImageTestWidget
from ui.diagnostics_panel.predict_training_set.predict_training_set_view import ImageTrainWidget
from ui.diagnostics_panel.predict_validation_set.predict_validation_set_view import ImageValidationWidget
from ui.diagnostics_panel.nn_internal_state.matrix_view import FirstLayerWeightsWidget
from ui.diagnostics_panel.nn_internal_state.matrix_view import FirstLayerGradientWidget
from ui.control_panel.control_panel_view import ControlWidget
from ui.dashboard_panel.cost.cost_view import CostWidget
from ui.dashboard_panel.accuracy.accuracy_view import AccuracyWidget
from ui.abstract.base_plot import PLOT_COST_HEIGHT, PLOT_COST_WIDTH

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.model = nn(self.config)
        self.setWindowTitle("digit classification")

        self.output_tabs = QTabWidget()
        self.image_test = ImageTestWidget(parent=self)
        self.image_train = ImageTrainWidget(parent=self)
        self.image_validation = ImageValidationWidget(parent=self)

        self.weights0_panel = FirstLayerWeightsWidget(parent=self)
        self.gradient0_panel = FirstLayerGradientWidget(parent=self)
        matrix_container = QWidget()
        layout_widget = QHBoxLayout()
        layout_widget.addWidget(self.weights0_panel)
        layout_widget.addWidget(self.gradient0_panel)
        matrix_container.setLayout(layout_widget)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(matrix_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(800)
        self.layout = QGridLayout()

        self.output_tabs.addTab(self.scroll_area, 'first non-input layer')
        self.output_tabs.addTab(self.image_train, 'training set')
        self.output_tabs.addTab(self.image_validation, 'validation set')
        self.output_tabs.addTab(self.image_test, 'test set')

        self.control_panel = ControlWidget(parent=self)
        self.cost_panel = CostWidget(parent=self,
                                     ylabel='cost', xlabel='epochs', ylim=[0, 3])
        self.accuracy_panel = AccuracyWidget(parent=self, ylabel='accuracy(correct / total)', xlabel='epochs', ylim=[0,1.1])
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setFixedHeight(PLOT_COST_HEIGHT)
        self.plot_tabs.setFixedWidth(PLOT_COST_WIDTH)
        self.plot_tabs.addTab(self.accuracy_panel, 'accuracy')
        self.plot_tabs.addTab(self.cost_panel, 'cost')

        layout_left = QVBoxLayout()
        layout_left.addWidget(self.plot_tabs, alignment=Qt.AlignHCenter)
        layout_left.addWidget(self.control_panel, alignment=Qt.AlignHCenter)
        layout_left.addStretch()

        layout_right = QVBoxLayout()
        layout_right.addWidget(self.output_tabs)
        layout_main = QHBoxLayout()
        layout_main.addLayout(layout_left)
        layout_main.addLayout(layout_right)
        main_container = QWidget()
        main_container.setLayout(layout_main)
        self.setCentralWidget(main_container)
        # self.show()
