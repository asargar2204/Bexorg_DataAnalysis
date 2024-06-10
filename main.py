from tkinter import messagebox
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLineEdit, QWidget, QLabel  # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import time
import sys

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.plot()

    def plot(self, x_data=None, y_data=None):
        self.axes.clear()
        if x_data is not None and y_data is not None:
            self.axes.plot(x_data, y_data, 'r-')
        self.draw()

class PlotWorker(QThread):
    data_signal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, amplitude, offset, frequency):
        super().__init__()
        self.amplitude = amplitude
        self.offset = offset
        self.frequency = frequency
        self.running = False
        self.data = []

    def run(self):
        self.running = True
        start_time = time.time()
        last_save_time = start_time
        while self.running:
            current_time = time.time() - start_time
            self.data.append((current_time, self.amplitude * np.sin(self.frequency * current_time) + self.offset))
            x_data, y_data = zip(*self.data)
            self.data_signal.emit(np.array(x_data), np.array(y_data))

            if time.time() - last_save_time >= 60:
                self.save_data()
                last_save_time = time.time()

            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.save_data()

    def save_data(self):
        timestamp = int(time.time())
        filename = f'data_{timestamp}.npy'
        np.save(filename, self.data)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Real-Time Data Acquisition and Visualization')

        self.start_stop_button = QPushButton('Start/Stop Plotting', self)
        self.amplitude_input = QLineEdit(self)
        self.offset_input = QLineEdit(self)
        self.frequency_input = QLineEdit(self)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Amplitude:'))
        layout.addWidget(self.amplitude_input)
        layout.addWidget(QLabel('Offset:'))
        layout.addWidget(self.offset_input)
        layout.addWidget(QLabel('Frequency:'))
        layout.addWidget(self.frequency_input)
        layout.addWidget(self.start_stop_button)

        self.plot_canvas = PlotCanvas(self)
        layout.addWidget(self.plot_canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.plot_worker = None
        self.start_stop_button.clicked.connect(self.toggle_plotting)

    def toggle_plotting(self):
        try:
            amplitude = float(self.amplitude_input.text())
            offset = float(self.offset_input.text())
            frequency = float(self.frequency_input.text())
        except ValueError:
            messagebox.warning(self, 'Invalid Input', 'Please enter valid numerical values for amplitude, offset, and frequency.')
            return

        if self.plot_worker is None or not self.plot_worker.isRunning():
            self.plot_worker = PlotWorker(amplitude, offset, frequency)
            self.plot_worker.data_signal.connect(self.plot_canvas.plot)
            self.plot_worker.start()
        else:
            self.plot_worker.stop()

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

