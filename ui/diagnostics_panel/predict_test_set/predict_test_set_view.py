from PyQt5.QtWidgets import QWidget

from ui.abstract.base_image import ImageWidget


class ImageTestWidget(ImageWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.previous_fail_button.hide()
        self.next_fail_button.hide()
        self.previous_fail_button.setEnabled(False)
        self.next_fail_button.setEnabled(False)
        self.labels = None
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
