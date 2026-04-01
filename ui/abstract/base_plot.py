import os
import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT

PLOT_COST_WIDTH = 450
PLOT_COST_HEIGHT = 400


class NavigationToolbarWhiteIcon(NavigationToolbar2QT):

    def white_image(mpl_icon_file_id):
        image_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'images')
        return os.path.abspath(os.path.join(image_dir, mpl_icon_file_id))

    toolitems = (
        ('Home', 'Reset original view', white_image('home'), 'home'),
        # ('Back', 'Back to previous view', white_image('back'), 'back'),
        # ('Forward', 'Forward to next view', white_image('forward'), 'forward'),
        (None, None, None, None),
        ('Pan',
         'Left button pans, Right button zooms\n'
         'x/y fixes axis, CTRL fixes aspect',
         white_image('move'), 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', white_image('zoom_to_rect'), 'zoom'),
        # ('Subplots', 'Configure subplots', white_image('subplots'), 'configure_subplots'),
        # ("Customize", "Edit axis, curve and image parameters",
         # white_image("qt4_editor_options"), "edit_parameters"),
        (None, None, None, None),
        ('Save', 'Save the figure', white_image('filesave'), 'save_figure'),
      )

    def __init__(self, canvas, parent=None, coordinates=True):
        super().__init__(canvas, parent, coordinates)


class PlotLineWidget(QWidget):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self.kwargs = kwargs
        epochs = self.parent().control_panel.config_panel.config.epochs
        self.layout_top = QHBoxLayout()
        self.layout = QVBoxLayout()
        self.canvas = FigureCanvas()
        self.ax = self.canvas.figure.add_subplot(111)
        self.init_axes(epochs)
        self.toolbar = NavigationToolbarWhiteIcon(canvas=self.canvas, parent=self)
        self.layout_top.addWidget(self.toolbar)
        self.layout.addLayout(self.layout_top)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.setFixedHeight(PLOT_COST_HEIGHT)
        self.setFixedWidth(PLOT_COST_WIDTH)

    def init_axes(self, epochs):
        self.ax.clear()
        self.ax.set_xlim([0, epochs])
        self.ax.set_ylim(self.kwargs['ylim'])
        # self.ax.set_yscale('log')
        self.ax.grid(True)
        self.ax.grid(which="minor")
        self.ax.set(xlabel=self.kwargs.get('xlabel'), ylabel=self.kwargs.get('ylabel'))
        self.canvas.figure.set_layout_engine('constrained')
        x = np.arange(1, epochs + 1)
        y = np.full(epochs, None)
        self.line_train, = self.ax.plot(x, y, label='training')
        self.line_validation, = self.ax.plot(x, y, label='validation')
        self.ax.legend()

    def update_plot(self, model):
        print(f'PlotLineWidget.update_plot not implemented')
