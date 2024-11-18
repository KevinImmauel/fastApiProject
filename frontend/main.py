import sys
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap


class ImagePredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the window
        self.setWindowTitle('Image Predictor')
        self.setGeometry(100, 100, 400, 300)

        # Layout setup
        layout = QVBoxLayout()

        # Image label
        self.image_label = QLabel('No image selected', self)
        layout.addWidget(self.image_label)

        # Button to select image
        self.select_button = QPushButton('Select Image', self)
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        # Button to make prediction
        self.predict_button = QPushButton('Predict', self)
        self.predict_button.clicked.connect(self.predict_image)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

        # Path to the selected image
        self.image_path = None

    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.jpeg)',
                                                   options=options)
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def predict_image(self):
        if self.image_path:
            url = 'http://127.0.0.1:8000/predict/'
            files = {'file': open(self.image_path, 'rb')}
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                print("Prediction Results:", result)
                # Show the prediction in a simple dialog box or label (customize as needed)
                self.image_label.setText(f"Predictions: {result}")
            else:
                self.image_label.setText(f"Error: {response.status_code}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImagePredictorApp()
    ex.show()
    sys.exit(app.exec_())
