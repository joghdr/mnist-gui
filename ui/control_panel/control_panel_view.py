import time

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from PyQt5.QtCore import Qt, pyqtSlot, QThread, QObject, pyqtSignal


from ui.abstract.button import Button
from ui.control_panel.digit_selection.digit_selection_view import DigitWidget
from ui.control_panel.settings.settings_view import ConfigWidget

CONTROL_PANEL_WIDTH = 500
CONTROL_PANEL_HEIGHT = 550

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


class ActionsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QHBoxLayout()
        self.train_panel = TrainWidget(parent=self)
        self.layout.addWidget(self.train_panel)
        self.setLayout(self.layout)


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

