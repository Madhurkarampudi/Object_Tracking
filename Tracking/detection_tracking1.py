import sys
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer

from PyQt5.QtGui import QImage, QPixmap
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label_video = QLabel(self)
        self.label_video.setGeometry(10, 10, 640, 480)

        button_load = QPushButton('Load Video', self)
        button_load.move(10, 500)
        button_load.clicked.connect(self.load_video)

        button_start = QPushButton('Start Tracking', self)
        button_start.move(110, 500)
        button_start.clicked.connect(self.start_tracking)

        button_stop = QPushButton('Stop Tracking', self)
        button_stop.move(220, 500)
        button_stop.clicked.connect(self.stop_tracking)

        self.tracker = None

    def load_video(self):
        # Open a file dialog to select a video file
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video files (*.mp4 *.avi)')
        if file_name:
            self.video = cv2.VideoCapture(file_name)
            if not self.video.isOpened():
                QMessageBox.critical(self, 'Error', 'Could not open video file')
                return
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(1000 // self.fps)

    def start_tracking(self):
        if self.tracker is not None:
            QMessageBox.warning(self, 'Warning', 'Tracker is already running')
            return
        if not hasattr(self, 'video') or self.video is None:
            QMessageBox.warning(self, 'Warning', 'Video not loaded')
            return
        # Initialize the tracker with a bounding box selected by the user
        bbox = cv2.selectROI(self.label_video.pixmap().toImage())
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(self.video.read()[1], bbox)

    def update_frame(self):
        if self.tracker is not None:
            ok, frame = self.video.read()
            if ok:
                ok, bbox = self.tracker.update(frame)
                if ok:
                    (x, y, w, h) = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    QMessageBox.warning(self, 'Warning', 'Tracking failure')
                    self.stop_tracking()
        else:
            ok, frame = self.video.read()
        if ok:
            height, width, bytes_per_line = frame.shape
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label_video.setPixmap(pixmap)
            self.label_video.setScaledContents(True)
        else:
            self.stop_tracking()

    def stop_tracking(self):
        # Stop the tracker
        self.tracker = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
