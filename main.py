import sys
import cv2
import numpy as np
import math
import xlsxwriter
from konvolusi import convolve as conv
from itertools import product
import csv
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

import os
import h5py
import imutils
from imutils import paths
from keras.applications import xception
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Mohamad Muqiit Faturrahman - 152018016
# Rizkika Siti Syifa - 152018030
# Siti Asy Syifa - 152018032

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('project.ui', self)
        self.image = None
        self.image_contrast = None
        self.image_output = None
        self.btnLoad.clicked.connect(self.loadClicked)
        self.btnSave.clicked.connect(self.saveClicked)
        self.btnIdentifikasi.clicked.connect(self.detectProcess)
        self.actionMean_Filter.triggered.connect(self.meanClicked)

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('fire_dataset/fire_images/fire.558.png')
        # flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image Files (*.png *.jpg *.jpeg)")
        # if flname:
        #     self.loadImage(flname)
        # else:
        #     print('Invalid Image')

    def loadImage(self, flname):
        self.image = cv2.imread(flname, cv2.IMREAD_COLOR)
        img = self.image
        # self.exportCSV(img, 'array_image')
        self.displayImage()

    def contrast(self, img):
        contrast = 1.4
        h, w = img.shape[:2]
        for i in np.arange(h):
            for j in np.arange(w):
                a = img.item(i, j)
                b = math.ceil(a * contrast)
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                img.itemset((i, j), b)
        self.image_contrast = img
        cv2.imshow("Contrast", self.image_contrast)

    def meanClicked(self):
        img = self.image
        kernel = np.ones((3, 3), np.float32) / 9
        mean = cv2.filter2D(self.image, -1, kernel)
        self.image = mean
        cv2.imshow("Mean", mean)
        cv2.imshow("Original Image", img)

    def detectProcess(self):
        print("Process")
        print("[INFO] loading model...")
        model = load_model("model/model.h5")

        # grab the paths to the fire and non-fire images, respectively
        print("[INFO] predicting...")
        firePaths = list(paths.list_images(os.path.sep.join(["fire_dataset", "fire_images"])))
        nonFirePaths = list(paths.list_images(os.path.sep.join(["fire_dataset", "non_fire_images"])))

        # combine the two image path lists, randomly shuffle them, and sample
        # them
        imagePaths = firePaths + nonFirePaths
        random.shuffle(imagePaths)
        imagePaths = imagePaths[:50]

        # loop over the sampled image paths
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and clone it
            image = cv2.imread(imagePath)
            output = image.copy()
            # resize the input image to be a fixed 128x128 pixels, ignoring
            # aspect ratio
            image = cv2.resize(image, (128, 128))
            image = image.astype("float32") / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # make predictions on the image
            preds = model.predict(np.expand_dims(image, axis=0))[0]
            j = np.argmax(preds)
            CLASSES = ["Non-Fire", "Fire"]
            label = CLASSES[j]
            # draw the activity on the output frame
            text = label if label == "Non-Fire" else "WARNING! Fire!"
            output = imutils.resize(output, width=500)
            cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.25, (0, 255, 0), 5)

            # write the output image to disk
            filename = "{}.png".format(i)
            p = os.path.sep.join([os.path.sep.join(["output", "fire_images"]), filename])
            cv2.imwrite(p, output)


    def exportXLSX(self, array, flname):
        workbook = xlsxwriter.Workbook(str(flname) + '.xlsx')
        worksheet = workbook.add_worksheet()
        # array = self.image
        row = 0
        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)
        workbook.close()

    def exportCSV(self, array, flname):
        with open(str(flname) + '.csv', 'w', newline='') as f_output:
            csv_output = csv.writer(f_output)
            csv_output.writerow(["Image Name ", "R", "G", "B"])
            width, height = array.shape[:2]
            print(f'{array}, Width {width}, Height {height}')  # show
            # Read the details of each pixel and write them to the file
            csv_output.writerows([array, array[x, y]] for x, y in product(range(width), range(height)))

    def saveClicked(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\',
                                                     "Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.image)
        else:
            print('Error')

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(
            self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()
        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
        elif windows == 2:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))
            self.hasilLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.hasilLabel.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Main Window')
window.show()
sys.exit(app.exec_())
