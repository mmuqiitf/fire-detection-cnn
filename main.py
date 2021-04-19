import math
import sys
import cv2
from konvolusi import convolve as conv
import xlsxwriter
from itertools import product
import csv
import numpy as np
from scipy import signal
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
import h5py
from keras.applications import xception
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import datetime as dt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import os

plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('project.ui', self)
        self.image = None
        self.image_contrast = None
        self.image_output = None
        self.image_flname = None
        self.base_folder = 'fire-dataset'
        self.data_folder = 'fire_dataset'
        self.train_data_folder = 'fire_dataset/fire_images'
        self.test_date_folder = 'fire_dataset/non_fire_images'
        self.categories = ['fire_images', 'non_fire_images']
        self.len_categories = len(self.categories)
        self.image_count = {}
        self.train_data = []
        self.df = None
        self.INPUT_SIZE = 255
        self.X_train = None
        self.btnLoad.clicked.connect(self.loadClicked)
        self.btnSave.clicked.connect(self.saveClicked)
        self.btnIdentifikasi.clicked.connect(self.detectProcess)

    @pyqtSlot()
    def loadClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image Files (*.png *.jpg *.jpeg)")
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    def loadImage(self, flname):
        self.image = cv2.imread(flname, cv2.IMREAD_COLOR)
        self.image_flname = flname
        self.displayImage()

    def create_mask(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 250])
        upper_hsv = np.array([250, 255, 255])
        mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # image segmentation function
    def segment_image(self, image):
        mask = self.create_mask(image)
        output = cv2.bitwise_and(image, image, mask=mask)
        return output / 255

    # sharpen the image
    def sharpen_image(self, image):
        kernel = np.ones((3, 3), np.float32) / 9
        image_filter = cv2.filter2D(image, -1, kernel)
        return image_filter

    def detectProcess(self):
        img = image.load_img(self.image_flname, target_size=(
            self.INPUT_SIZE, self.INPUT_SIZE))
        # convert image to array
        img = image.img_to_array(img)
        print(img)
        # masking and segmentation
        image_segmented = self.segment_image(img)
        # sharpen
        image_sharpen = self.sharpen_image(image_segmented)
        x = xception.preprocess_input(
            np.expand_dims(image_sharpen.copy(), axis=0))
        loaded_model = load_model('./saved_model')
        xception_bf = xception.Xception(
            weights='imagenet', include_top=False, pooling='avg')
        bf_train_x = xception_bf.predict(x, batch_size=32, verbose=1)
        predictions = loaded_model.predict_classes(bf_train_x)
        if predictions == 0:
            print("Predictions : ", predictions, " is Fire!")
        elif predictions == 1:
            print("Predictions : ", predictions, " is Non-Fire")
        label = "Fire" if predictions == 0 else "Non-Fire"
        color = (0, 0, 255) if predictions == 0 else (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        thickness = 2
        cv2.putText(self.image, label, org, font, fontScale,
                    color, thickness, cv2.LINE_AA)
        self.displayImage(2)
        cv2.imshow("Image Masking", image_sharpen)
        cv2.waitKey(0)

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
