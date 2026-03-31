from ui.abstract.base_plot import PlotLineWidget


class CostWidget(PlotLineWidget):
    def __init__(self, parent, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def update_plot(self, model):
        self.line_train.set_ydata(model.C_train)
        self.line_validation.set_ydata(model.C_validation)
        self.canvas.draw_idle()
