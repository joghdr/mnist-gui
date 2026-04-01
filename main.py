#!/usr/bin/env python
import sys, os
from PyQt5.QtWidgets import QApplication
from ui.main_window_view import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    path_style = os.path.join(os.path.dirname(__file__), 'assets','style.css')
    path_style = os.path.abspath(path_style)

    try:
      with open(path_style, 'r') as stylefile:
          window.setStyleSheet(stylefile.read())
    except FileNotFoundError:
      print(f'Style file {path_style} not found')

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

