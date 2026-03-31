from PyQt5.QtWidgets import QWidget

from ui.abstract.base_image import ImageWidget


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

