import sys
import cv2
import numpy as np
import queue
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from mpi4py import MPI

class WorkerThread(QThread):
    processing_done = pyqtSignal(np.ndarray)

    def __init__(self, task_queue, rank):
        super().__init__()
        self.task_queue = task_queue
        self.rank = rank

    def run(self):
        while True:
            task = self.task_queue.get()

            if task is None:
                break

            image_path, operation = task
            result = self.process_image(image_path, operation)
            if result is not None:
                if self.rank == 0:
                    window.display_processed_image(result)
                else:
                    self.processing_done.emit(result)

    def process_image(self, image_path, operation):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if operation == 'Erosion':
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.erode(img, kernel, iterations=1)
        elif operation == 'Dilation':
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.dilate(img, kernel, iterations=1)
        elif operation == 'Opening':
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif operation == 'Closing':
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            print("Invalid operation:", operation)
            return None

        return result

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        buttons_layout = QHBoxLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedHeight(50)
        buttons_layout.addWidget(self.upload_button)

        self.operations_combo = QComboBox()
        self.operations_combo.addItems(['Erosion', 'Dilation', 'Opening', 'Closing'])
        self.operations_combo.setFixedHeight(50)
        buttons_layout.addWidget(self.operations_combo)

        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setFixedHeight(50)
        buttons_layout.addWidget(self.process_button)

        self.download_button = QPushButton("Download Result")
        self.download_button.clicked.connect(self.download_image)
        self.download_button.setFixedHeight(50)
        buttons_layout.addWidget(self.download_button)

        layout.addLayout(buttons_layout)

        self.central_widget.setLayout(layout)

        self.task_queue = queue.Queue()
        self.worker_thread = WorkerThread(self.task_queue, rank=0)
        self.worker_thread.start()

        self.selected_image_path = None

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        if self.rank != 0:
            self.worker_thread.processing_done.connect(self.display_processed_image)

    def upload_image(self):
        self.selected_image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        pixmap = QPixmap(self.selected_image_path)
        self.image_label.setPixmap(pixmap)

    def start_processing(self):
        if self.selected_image_path:
            self.task_queue.put((self.selected_image_path, self.operations_combo.currentText()))
        else:
            print("Please upload an image first.")

    def display_processed_image(self, result):
        if result is not None and result.size != 0:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            h, w, ch = result.shape
            bytes_per_line = ch * w
            q_img = QImage(result.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.result_label.setPixmap(pixmap)
        else:
            print("Error: Empty or None result received.")

    def download_image(self):
        if hasattr(self, 'result'):
            result = self.result
            if result is not None and result.size != 0:
                filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg)")
                if filename:
                    try:
                        cv2.imwrite(filename, result)
                        print("Image saved successfully!")
                    except Exception as e:
                        print("Error saving image:", e)
                else:
                    print("Save operation cancelled.")
            else:
                print("Error: Empty or None result received.")
        else:
            print("Please process the image first.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
