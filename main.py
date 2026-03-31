#!/usr/bin/env python
import sys
import time
import numpy as np


from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QListWidget,
    QTabWidget,
    QScrollArea,
)
from PyQt5.QtGui import QFont, QFontDatabase, QColor as Color
from PyQt5.QtCore import Qt, QSize, QObject, QRunnable, QThreadPool, QThread, pyqtSlot, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib as mpl
#### my modules


from core.Config import Config
from core.nn import nn
from core import plot_settings
from core.functions import logit, relu, cross_entropy, activation_dict, cost_dict
from ui.dashboard_panel.cost.cost_view import CostWidget
from ui.dashboard_panel.accuracy.accuracy_view import AccuracyWidget
from ui.abstract.base_plot import PLOT_COST_HEIGHT, PLOT_COST_WIDTH

PLOT_TEST_WIDTH = 650
PLOT_TEST_HEIGHT = 650
CONTROL_PANEL_WIDTH = 500
CONTROL_PANEL_HEIGHT = 550
WEIGHT_FILTER_WIDTH = 150
WEIGHT_FILTER_HEIGHT = 150

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

class InputTextWidget(QLineEdit):

    def __init__(self, text = '', parent=None):
        super().__init__(parent)
        self.setText(text)
        self.editingFinished.connect(self.clean)

    def clean(self):
        current = self.text()
        clean_text = current.replace(' ','_')
        self.setText(clean_text)

class RunIdWidget(QWidget):
    def __init__(self,  text='', parent=None):
        super().__init__(parent)
        label = QLabel('run id', alignment = Qt.AlignCenter)
        self.input_text = InputTextWidget(text=text, parent=self)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.input_text)
        self.setLayout(layout)

class Button(QPushButton):
    def __init__(self, label, run, parent=None, enabled=True):
        super().__init__(parent)
        self.setText(label)
        self.setFixedHeight(45)
        self.setEnabled(enabled)
        self.clicked.connect(run)

class PlotFilterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas()
        self.canvas.figure.set_layout_engine('constrained')
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_axis_off()
        image_template = np.random.sample((28, 28))
        self.heatmap = self.ax.matshow(image_template, cmap='cmap_image')
        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.setFixedSize(PLOT_TEST_WIDTH, PLOT_TEST_HEIGHT)

class PlotCostWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, func, *args, sleep=1):
        super().__init__()
        self.func = func
        self.args = args
        self.sleep = sleep
        self.wait = False
        self.done = False

    @pyqtSlot()
    def run(self):
        while not self.done:
            while self.wait:
                time.sleep(self.sleep)
            self.func(*self.args)
            self.wait = True
        self.finished.emit()

class TrainWorker(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, model):
        super().__init__()
        self.model = model

    @pyqtSlot()
    def run(self):
        self.model.train_setup()
        for i in self.model.training_iterator:
            # time.sleep(5)
            self.model.train_one_epoch(i)
            self.progress.emit(i)

            while self.model.pause_training:
                time.sleep(0.2)
                if self.model.end_training:
                        break
            if self.model.end_training:
                break
        self.finished.emit()

class TrainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = self.parent().parent().parent()
        self.label = QLabel('TRAIN', alignment = Qt.AlignCenter)
        self.layout = QGridLayout()
        self.start_button = Button('start', lambda: None, parent=self)
        self.pause_button = Button('pause', lambda: None, parent=self, enabled=False)
        self.end_button = Button('end', lambda: None, parent=self, enabled=False)
        self.connect_buttons()
        # self.pause_button.setEnabled(False)
        self.layout.addWidget(self.label, 0, 0, 1, 2)
        self.layout.addWidget(self.start_button , 1, 0, 1, 2)
        self.layout.addWidget(self.pause_button , 2, 0)
        self.layout.addWidget(self.end_button , 2, 1)
        self.setLayout(self.layout)

    def connect_buttons(self):
        #start button
        self.start_button.clicked.connect(self.disable_panel)
        self.start_button.clicked.connect(lambda: self.main_window.control_panel.config_panel.update_config_from_panel(force_weights=True))
        self.start_button.clicked.connect(self.main_window.image_train.init_data)
        self.start_button.clicked.connect(self.main_window.image_validation.init_data)
        self.start_button.clicked.connect(self.main_window.image_test.init_data)
        self.start_button.clicked.connect(self.main_window.weights0_panel.init)
        self.start_button.clicked.connect(self.main_window.gradient0_panel.init)

        self.start_button.clicked.connect(self.train_model)

        #pause button
        self.pause_button.clicked.connect(self.main_window.model.pause_continue_training)
        self.pause_button.clicked.connect(self.main_window.image_test.reset_predicted)
        self.pause_button.clicked.connect(self.main_window.image_train.reset_predicted)
        self.pause_button.clicked.connect(self.main_window.image_validation.reset_predicted)

        def pause_button_set_label_pause():
          self.pause_button.setText("pause")

        def pause_button_set_label_continue():
          self.pause_button.setText("continue")

        def pause_button_new_label():
          current_label = self.pause_button.text()
          if current_label == "pause":
            pause_button_set_label_continue()
            return
          pause_button_set_label_pause()

        def pause_action_on_images():
          if self.main_window.image_train.isEnabled():
            self.main_window.image_train.setEnabled(False)
          else:
            self.main_window.image_train.setEnabled(True)
          if self.main_window.image_validation.isEnabled():
            self.main_window.image_validation.setEnabled(False)
          else:
            self.main_window.image_validation.setEnabled(True)
          if self.main_window.image_test.isEnabled():
            self.main_window.image_test.setEnabled(False)
          else:
            self.main_window.image_test.setEnabled(True)

        self.pause_button.clicked.connect(pause_button_new_label)
        self.pause_button.clicked.connect(pause_action_on_images)

        #end button
        output_tabs = self.main_window.output_tabs

        self.end_button.clicked.connect(self.main_window.model.end)
        self.end_button.clicked.connect(self.end_plots)
        self.end_button.clicked.connect(pause_button_set_label_pause)




    def train_model(self):
        start = time.time()
        self.counter = 0
        model = self.main_window.model


        self.thread_train = QThread()
        self.worker_train = TrainWorker(model)
        self.worker_train.moveToThread(self.thread_train)

        self.thread_plot_cost = QThread()
        self.worker_plot_cost = PlotCostWorker(self.update_cost_accuracy_plots, sleep=0.2)
        self.worker_plot_cost.moveToThread(self.thread_plot_cost)

        self.thread_plot_weights = QThread()
        self.worker_plot_weights = PlotCostWorker(self.update_weight_plots, sleep=0.2)
        self.worker_plot_weights.moveToThread(self.thread_plot_weights)

        self.thread_train.started.connect(self.worker_train.run)
        self.thread_train.finished.connect(self.worker_train.deleteLater)
        self.thread_train.finished.connect(self.thread_train.deleteLater)
        self.thread_train.finished.connect(self.enable_panel)
        self.thread_train.finished.connect(self.main_window.image_train.reset_predicted)
        self.thread_train.finished.connect(self.main_window.image_validation.reset_predicted)
        self.thread_train.finished.connect(self.main_window.image_test.reset_predicted)
        self.thread_train.finished.connect(lambda: print(f'------> thread_train finished, time elapsed = {time.time() - start}'))

        self.thread_plot_cost.started.connect(self.worker_plot_cost.run)
        self.thread_plot_cost.finished.connect(self.worker_plot_cost.deleteLater)
        self.thread_plot_cost.finished.connect(self.thread_plot_cost.deleteLater)
        self.thread_plot_cost.finished.connect(lambda: print(f'-----------> thread_plot_cost finished, plot was refreshed {self.counter} times'))

        self.thread_plot_weights.started.connect(self.worker_plot_weights.run)
        self.thread_plot_weights.finished.connect(self.worker_plot_weights.deleteLater)
        self.thread_plot_weights.finished.connect(self.thread_plot_weights.deleteLater)
        self.thread_plot_weights.finished.connect(lambda: print(f'-----------> thread_plot_weights finished, plot was refreshed {self.counter} times'))


        self.worker_train.progress.connect(self.thread_plot_cost.start)
        self.worker_train.progress.connect(self.thread_plot_weights.start)
        self.worker_train.progress.connect(self.refresh_plots)
        self.worker_train.finished.connect(self.thread_train.quit)
        self.worker_train.finished.connect(self.end_plots)

        self.worker_plot_cost.finished.connect(self.thread_plot_cost.quit)

        self.worker_plot_weights.finished.connect(self.thread_plot_weights.quit)

        self.thread_train.start()

    def update_cost_accuracy_plots(self):
        model = self.main_window.model
        update_cost = self.main_window.cost_panel.update_plot
        update_accuracy = self.main_window.accuracy_panel.update_plot
        update_cost(model)
        update_accuracy(model)

    def update_weight_plots(self):
        self.main_window.weights0_panel.update_quadmesh()
        self.main_window.gradient0_panel.update_quadmesh()


    @pyqtSlot()
    def refresh_plots(self):
        self.counter += 1
        self.worker_plot_cost.wait = False
        self.worker_plot_weights.wait = False

    @pyqtSlot()
    def end_plots(self):
        self.worker_plot_cost.wait = False
        self.worker_plot_cost.done = True
        self.worker_plot_weights.wait = False
        self.worker_plot_weights.done = True

    def disable_panel(self):
        config = self.main_window.config
        model = self.main_window.model
        cost_panel = self.main_window.cost_panel
        control_panel = self.main_window.control_panel
        actions_panel = control_panel.actions_panel
        digit_panel = control_panel.digit_panel
        config_panel = control_panel.config_panel
        config_panel.setEnabled(False)
        digit_panel.setEnabled(False)
        self.main_window.image_train.setEnabled(False)
        self.main_window.image_validation.setEnabled(False)
        self.main_window.image_test.setEnabled(False)
        actions_panel.train_panel.start_button.setEnabled(False)
        actions_panel.train_panel.pause_button.setEnabled(True)
        actions_panel.train_panel.end_button.setEnabled(True)

    def enable_panel(self):
        config = self.main_window.config
        model = self.main_window.model
        cost_panel = self.main_window.cost_panel
        control_panel = self.main_window.control_panel
        actions_panel = control_panel.actions_panel
        digit_panel = control_panel.digit_panel
        config_panel = control_panel.config_panel
        config_panel.setEnabled(True)
        digit_panel.setEnabled(True)
        self.main_window.image_train.setEnabled(True)
        self.main_window.image_validation.setEnabled(True)
        self.main_window.image_test.setEnabled(True)
        actions_panel.train_panel.start_button.setEnabled(True)
        actions_panel.train_panel.pause_button.setEnabled(False)
        actions_panel.train_panel.end_button.setEnabled(False)

class ImageWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.main_window = self.parent()
        self.X = None
        self.labels = None
        self.slices = None
        self.Images = None
        self.predictions = None
        self.iterator = 0
        self.loop_counter = 0
        self.display = PlotFilterWidget(parent=self)
        self.display_layout = QVBoxLayout()
        self.display_layout.addWidget(self.display)
        self.timer = None

        self.next_button = Button('next', self.show_next, parent=self)
        self.next_fail_button = Button('next fail', self.show_next_fail, parent=self)
        self.previous_button = Button('previous', self.show_previous, parent=self)
        self.previous_fail_button = Button('previous fail', self.show_previous_fail, parent=self)

        self.actions_layout = QGridLayout()
        self.actions_layout.addWidget(self.previous_fail_button, 0, 0, 2, 2)
        self.actions_layout.addWidget(self.previous_button,      0, 2, 2, 2)
        self.actions_layout.addWidget(self.next_button,          0, 4, 2, 2)
        self.actions_layout.addWidget(self.next_fail_button,     0, 6, 2, 2)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.display_layout)
        self.layout.addLayout(self.actions_layout)
        self.setLayout(self.layout)


    def init_data(self):
        print(f'function init_data not implemented in base class ImageWidget')

    def set_images(self):
        data = self.main_window.model.data
        self.Images = data.images(self.X)

    def set_predictions(self):
        model = self.main_window.model
        self.predictions = model.predict(self.X)

    def get_index(self):
        if self.Images is None:
            self.set_images()
        return self.iterator % len(self.Images)

    def get_title(self):
        index = self.get_index()
        return (f'Image # {self.get_index() + 1} / {len(self.Images)}: '
                f'Prediction: {self.predictions[index]} '
                f'True value: {self.labels[index]}')

    def current_is_correct(self):
        return self.predictions[self.get_index()] == self.labels[self.get_index()]

    def show_current(self):
        if self.Images is None or self.predictions is None:
            self.set_images()
            self.set_predictions()
        index = self.get_index()
        self.display.heatmap.set_data(self.Images[index])
        self.display.ax.set_title(self.get_title())
        self.display.canvas.draw()

    def show_next(self, check_timer=False):
        if self.Images is None or self.predictions is None:
            self.set_images()
            self.set_predictions()
        self.iterator += 1
        index = self.get_index()
        self.display.heatmap.set_data(self.Images[index])
        self.display.ax.set_title(self.get_title())
        self.display.canvas.draw()

        if check_timer:
            self.loop_counter += 1
            if self.loop_counter == len(self.Images):
                self.timer.stop()
                print(f'no failures were found')
            elif not self.current_is_correct():
                self.timer.stop()

    def show_previous(self, check_timer=False):
        self.iterator -= 2
        self.show_next(check_timer)

    def show_next_fail(self):
        self.loop_counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.show_next(check_timer=True))
        self.timer.start(1)

    def show_previous_fail(self):
        self.loop_counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.show_previous(check_timer=True))
        self.timer.start(1)

    def reset_predicted(self):
        self.predictions = None

    def test_extended(self):
        print(f'function test_extended in base class ImageWidget not implemented')

class ImageTestWidget(ImageWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.actions_layout.removeWidget(self.previous_fail_button)
        self.actions_layout.removeWidget(self.next_fail_button)
        self.actions_layout.addWidget(QWidget(), 0, 0, 2, 2)
        self.actions_layout.addWidget(QWidget(), 0, 6, 2, 2)
        del self.previous_fail_button
        del self.next_fail_button
        del self.labels
        self.init_data()

    def init_data(self):
        self.Images = None
        slices = self.main_window.model.data.slice_test
        self.X = self.main_window.model.data.X_test[slices]
        self.show_current()

    def get_title(self):
        index = self.get_index()
        return (f'Image # {self.get_index() + 1} / {len(self.Images)}: '
                f'Prediction: {self.predictions[index]}')

class ImageTrainWidget(ImageWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_data()

    def init_data(self):
        self.Images = None
        slices = self.main_window.model.data.slice_train_flat
        self.X = self.main_window.model.data.X[slices]
        self.labels = self.main_window.model.data.labels[slices[1]]
        self.show_current()

class ImageValidationWidget(ImageWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_data()

    def init_data(self):
        self.Images = None
        slices = self.main_window.model.data.slice_validation
        self.X = self.main_window.model.data.X[slices]
        self.labels = self.main_window.model.data.labels[slices[1]]
        self.show_current()

class FirstLayerMatrixWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = self.parent()
        self.matrix = None
        self.figure =Figure(figsize=(20,20))
        self.canvas = FigureCanvas(self.figure)
        self.figure.set_layout_engine('constrained')
        self.axes = None
        self.quadmesh = None
        self.subfigs = None
        self.n_of_subfigs = None
        self.n_of_filters = None
        self.n_of_rows = None
        self.quadmesh_template = None
        self.layer_shape = None
        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.init()

    def set_layer_shape(self):
        config = self.main_window.config
        nodes = self.main_window.model.grids_train._Grids__mknodes(config).astype(list)
        shape = (nodes[1], nodes[0] + 1)
        self.layer_shape = shape

    def init(self):
        self.set_layer_shape()
        self.n_of_subfigs = 2
        self.n_of_filters = self.layer_shape[0]
        self.n_of_rows = -(self.n_of_filters // -self.n_of_subfigs)
        self.canvas.setFixedSize(WEIGHT_FILTER_WIDTH * self.n_of_subfigs, WEIGHT_FILTER_HEIGHT * self.n_of_rows)
        self.canvas.figure.clear()
        self.subfigs = self.canvas.figure.subfigures(1, self.n_of_subfigs)
        self.setup_axes()

    def setup_axes(self):
        self.axes = np.array([])
        self.quadmesh = []
        delta = 0.000001
        count = 0

        for subfig in self.subfigs:
            self.axes = np.append(self.axes, subfig.subplots(self.n_of_rows, 1))
        self.axes.flatten()

        for ax in self.axes:
            count += 1
            self.quadmesh_template = np.random.sample(self.layer_shape[1] + 27 ).reshape(28, 29)
            maximum = np.max(np.abs(self.quadmesh_template[:,1:]))
            quadmesh = ax.pcolormesh(self.quadmesh_template, cmap='cmap_weights', vmin = -maximum - delta, vmax = maximum + delta, linewidth=0, rasterized=True)
            ax.yaxis.set_inverted(True)
            ax.set_axis_off()
            ax.set_title(f'({count})', size=10, loc='left', y=1, pad=-10)
            self.quadmesh.append(quadmesh)
        self.zero_quadmesh()

    def set_matrix(self):
        print(f'set_matrix not implemented in base class FirstLayerMatrixWidget')


    def get_filter_array(self, w_row):
        self.set_matrix()
        weights = w_row[1:]
        bias = w_row[0]
        filter_array = np.zeros_like(self.quadmesh_template)
        filter_array[:,1:] = weights.reshape(28, 28)
        filter_array[:,0] = bias
        return filter_array

    def zero_quadmesh(self):
        for i in range(self.n_of_subfigs * self.n_of_rows):
            array = np.zeros((28, 29))
            self.quadmesh[i].set_array(array)
        self.canvas.draw_idle()

    def update_quadmesh(self):
        self.set_matrix()
        delta = 0.000001
        vmax = np.max(np.abs(self.matrix)) + delta
        vmin = -vmax
        for i in range(len(self.matrix)):
            array = self.get_filter_array(self.matrix[i,:])
            vmax_i = np.max(np.abs(array))
            vmin_i = vmax_i

            vmax_i = 0.1
            vmin_i = -vmax_i
            self.quadmesh[i].set_clim(vmin, vmax)
            self.quadmesh[i].set_array(array)
        self.canvas.draw_idle()

class FirstLayerWeightsWidget(FirstLayerMatrixWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure.suptitle(f'Weights (node)', color=mpl.rcParams['axes.titlecolor'], size=11)

    def init(self):
        super().init()
        self.figure.suptitle(f'Weights (node)', color=mpl.rcParams['axes.titlecolor'], size=11)

    def set_matrix(self):
        self.matrix = self.main_window.model.grids_train.W[0]

class FirstLayerGradientWidget(FirstLayerMatrixWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure.suptitle(f'Gradients (node)', color=mpl.rcParams['axes.titlecolor'], size=11)

    def init(self):
        super().init()
        self.figure.suptitle(f'Gradients (node)', color=mpl.rcParams['axes.titlecolor'], size=11)


    def set_matrix(self):
        self.matrix = self.main_window.model.grids_train.Gradient_W[0]

class InspectWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = self.parent().parent().parent()
        self.label = QLabel('INSPECT', alignment = Qt.AlignCenter)
        self.layout = QGridLayout()
        self.start_button = Button('weights', lambda: print(f'not implemented'), parent=self)
        self.next_button = Button('gradient', lambda: print(f'not implemented'), parent=self)
        self.layout.addWidget(self.label, 0, 0, 1, 2)
        self.layout.addWidget(self.start_button , 1, 0)
        self.layout.addWidget(self.next_button , 1, 1)
        self.setLayout(self.layout)

class ActionsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QHBoxLayout()
        self.train_panel = TrainWidget(parent=self)
        self.layout.addWidget(self.train_panel)
        self.setLayout(self.layout)

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

class ControlWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.actions_panel = ActionsWidget(parent=self)
        self.digit_panel = DigitWidget(parent=self)
        self.config_panel = ConfigWidget(parent=self)

        self.layout.addStretch()
        self.layout.addWidget(self.actions_panel)
        self.layout.addWidget(self.digit_panel)
        self.layout.addWidget(self.config_panel)
        self.setLayout(self.layout)
        self.setFixedHeight(CONTROL_PANEL_HEIGHT)
        self.setFixedWidth(CONTROL_PANEL_WIDTH)

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
        widget = QWidget()
        layout_widget = QHBoxLayout()
        layout_widget.addWidget(self.weights0_panel)
        layout_widget.addWidget(self.gradient0_panel)
        widget.setLayout(layout_widget)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(widget)
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
        widget = QWidget()
        widget.setLayout(layout_main)
        self.setCentralWidget(widget)
        self.show()

app = QApplication(sys.argv)
window = MainWindow()
with open('.style.css', 'r') as stylefile:
    window.setStyleSheet(stylefile.read())
app.exec()
