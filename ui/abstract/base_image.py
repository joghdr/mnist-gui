
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvas
from ui.abstract.button import Button
from ui.abstract import plot_settings


PLOT_TEST_WIDTH = 650
PLOT_TEST_HEIGHT = 650


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
