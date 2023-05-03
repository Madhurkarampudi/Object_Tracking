import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from imutils.video import VideoStream
import time

class CamShiftTracker(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = VideoStream(src=0).start()
        time.sleep(1.0)
        frame = self.cap.read()
        bbox = cv2.selectROI(frame)
        x, y, w, h = bbox
        self.track_window = (x, y, w, h)
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.parameters = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # Create GUI elements
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
        self.start_btn = QPushButton('Start', self)
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn = QPushButton('Stop', self)
        self.stop_btn.clicked.connect(self.stop_tracking)

        # Add elements to layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        self.tracking = False

    def start_tracking(self):
        self.tracking = True
        while self.tracking:
            frame = self.cap.read()
            ok = not frame is None
            if ok == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

                ok, self.track_window = cv2.CamShift(dst, self.track_window, self.parameters)

                pts = cv2.boxPoints(ok)
                pts = np.int0(pts)
                img2 = cv2.polylines(frame, [pts], True, 255, 2)

                # Convert image to QImage and display on label
                img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(pixmap)

                if cv2.waitKey(1) == 13:
                    break
            else:
                break

    def stop_tracking(self):
        self.tracking = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tracker = CamShiftTracker()
    tracker.show()
    sys.exit(app.exec_())
