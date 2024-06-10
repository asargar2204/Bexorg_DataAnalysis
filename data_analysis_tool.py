import os
import shutil
import statistics
import sys
import tempfile

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QComboBox, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.fft import fft

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, datasets, plot_type='line', x_label='', y_label=''):
        self.axes.clear()
        for x_data, y_data, label in datasets:
            if plot_type == 'line':
                self.axes.plot(x_data, y_data, label=label)
            elif plot_type == 'hist':
                self.axes.hist(y_data, bins=30, alpha=0.5, label=label)
            elif plot_type == 'fft':
                N = len(y_data)
                T = x_data[1] - x_data[0] if len(x_data) > 1 else 1
                yf = fft(y_data)
                xf = np.fft.fftfreq(N, T)[:N // 2]
                self.axes.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), label=label)
            elif plot_type == 'regression':
                self.axes.plot(x_data, y_data, 'o', label=label)
                m, b = np.polyfit(x_data, y_data, 1)
                self.axes.plot(x_data, m * np.array(x_data) + b, '-')

        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        if any([line.get_label() != '_nolegend_' for line in self.axes.get_lines()]):
            self.axes.legend()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Comprehensive Data Analysis Tool')
        self.setGeometry(100, 100, 1200, 800)

        self.file_picker_button = QPushButton('Load .npy Files', self)
        self.file_picker_button.clicked.connect(self.load_files)

        self.plot_canvas = PlotCanvas(self)

        self.plot_type_selector = QComboBox(self)
        self.plot_type_selector.addItems(['Select Plot', 'line', 'hist', 'fft', 'regression'])
        self.plot_type_selector.currentIndexChanged.connect(self.update_plot)

        self.dataset_selector = QListWidget(self)
        self.dataset_selector.setGeometry(50, 50, 200, 400)
        self.dataset_selector.itemChanged.connect(self.update_plot)

        self.stats_table = QTableWidget(self)
        self.stats_table.setRowCount(0)
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Dataset", "Mean", "Median", "Std Dev"])

        self.save_button = QPushButton('Save Report', self)
        self.save_button.clicked.connect(self.save_pdf)

        layout = QVBoxLayout()
        layout.addWidget(self.file_picker_button)
        layout.addWidget(self.plot_canvas)
        layout.addWidget(self.plot_type_selector)
        layout.addWidget(QLabel("Select Dataset(s):"))
        layout.addWidget(self.dataset_selector)
        layout.addWidget(QLabel("Dataset Statistics:"))
        layout.addWidget(self.stats_table)
        layout.addWidget(self.save_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.datasets = {}

    def load_files(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open Data Files", "", "Data Files (*.npy)")
        for file_name in file_names:
            short_name = os.path.basename(file_name)
            item = QListWidgetItem(short_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.dataset_selector.addItem(item)
            data = np.load(file_name, allow_pickle=True)
            self.datasets[short_name] = data
        self.update_plot()

    def update_plot(self):
        datasets = []
        for index in range(self.dataset_selector.count()):
            item = self.dataset_selector.item(index)
            if item.checkState() == Qt.Checked:
                name = item.text()
                if name in self.datasets:
                    data = self.datasets[name]
                    x_data, y_data = zip(*data)
                    datasets.append((np.array(x_data), np.array(y_data), name))

        plot_type = self.plot_type_selector.currentText()
        x_label = 'Time' if plot_type != 'fft' else 'Frequency'
        y_label = 'Amplitude' if plot_type != 'hist' else 'Frequency'
        if plot_type != 'Select Plot':
            self.plot_canvas.plot(datasets, plot_type, x_label, y_label)
            self.update_stats_table(datasets)

    def update_stats_table(self, datasets):
        self.stats_table.setRowCount(len(datasets))
        for i, (_, y_data, name) in enumerate(datasets):
            mean = np.mean(y_data)
            median = statistics.median(y_data)
            std_dev = np.std(y_data)
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{mean:.2f}"))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{median:.2f}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{std_dev:.2f}"))
    def save_pdf(self):
        temp_dir = tempfile.mkdtemp()
        pdf_file_path = os.path.join(temp_dir, 'analysis_report.pdf')
        pdf = PdfPages(pdf_file_path)

        try:
            datasets = []
            # Gather all selected datasets
            for index in range(self.dataset_selector.count()):
                item = self.dataset_selector.item(index)
                if item.checkState() == Qt.Checked:
                    name = item.text()
                    if name in self.datasets:
                        data = self.datasets[name]
                        x_data, y_data = zip(*data)
                        datasets.append((np.array(x_data), np.array(y_data), name))

            # Create a statistics summary page
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, frame_on=False)
            ax.axis('off')
            ypos = 0.9
            plt.text(0.05, ypos, "Detailed Statistics", fontsize=16, weight='bold')
            for x_data, y_data, name in datasets:
                mean = np.mean(y_data)
                median = np.median(y_data)
                std_dev = np.std(y_data)
                ypos -= 0.1
                plt.text(0.05, ypos, f"{name}: Mean = {mean:.2f}, Median = {median:.2f}, Std Dev = {std_dev:.2f}", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

            # Generate plots for each plot type
            plot_types = ['line', 'hist', 'fft', 'regression']
            for plot_type in plot_types:
                fig, ax = plt.subplots(figsize=(8, 6))
                # Plot each dataset
                for x_data, y_data, name in datasets:
                    if plot_type == 'line':
                        ax.plot(x_data, y_data, label=name)
                    elif plot_type == 'hist':
                        ax.hist(y_data, bins=30, alpha=0.5, label=name)
                    elif plot_type == 'fft':
                        N = len(y_data)
                        T = x_data[1] - x_data[0] if len(x_data) > 1 else 1
                        yf = fft(y_data)
                        xf = np.fft.fftfreq(N, T)[:N // 2]
                        ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), label=name)
                    elif plot_type == 'regression':
                        m, b = np.polyfit(x_data, y_data, 1)
                        ax.plot(x_data, y_data, 'o', label=f'{name} Data Points')
                        ax.plot(x_data, m * np.array(x_data) + b, '-', label=f'{name} Fit')

                ax.set_title(f"Combined {plot_type.capitalize()} Plot")
                ax.set_xlabel('Time' if plot_type != 'fft' else 'Frequency')
                ax.set_ylabel('Amplitude' if plot_type != 'hist' else 'Frequency')
                ax.legend()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            pdf.close()
            save_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
            if save_path:
                shutil.move(pdf_file_path, save_path)
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
