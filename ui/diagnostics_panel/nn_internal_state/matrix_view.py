
import numpy as np
import matplotlib as mpl
from PyQt5.QtWidgets import QWidget, QGridLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas

WEIGHT_FILTER_WIDTH = 150
WEIGHT_FILTER_HEIGHT = 150


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
