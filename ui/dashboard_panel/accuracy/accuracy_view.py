from ui.abstract.base_plot import PlotLineWidget


class AccuracyWidget(PlotLineWidget):
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.ax.legend(loc='lower right')

    def update_plot(self, model):
        self.line_train.set_ydata(model.Acc_train)
        self.line_validation.set_ydata(model.Acc_validation)
        self.ax.legend(loc='lower right')
        self.canvas.draw_idle()
