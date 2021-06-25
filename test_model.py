import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
model = keras.models.load_model('model.ckpt')
# name_test = './DATA_TEST/5_5.png'
# img_test = cv2.imread(name_test)
# img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
# img_test = cv2.resize(src=img_test, dsize=(28, 28))
# plt.imshow(img_test)
# y_predict = model.predict(img_test.reshape(1, 28, 28, 1))
# print('Giá trị dự đoán: ', np.argmax(y_predict))
# plt.show()


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def max_6(coutours):
    (coutourss, boundingBoxes) = sort_contours(coutours)
    S = np.ones(len(coutours))
    S_xywh = np.full((len(coutours), 4), 0)
    Max_six = np.full((12, 4), 0)
    j = 0
    for c in coutourss:
        x, y, w, h = cv2.boundingRect(c)
        S[j] = (w+h)*2
        S_xywh[j] = [x, y, w, h]
        j += 1
    S[0] = 1
    for i in range(6):
        S[np.where(S == max(S))] = 0
    m = 0
    for i in range(len(S)):
        if(S[i] == 0):
            Max_six[m] = S_xywh[i]
            m += 1
    return Max_six


def convert_back(predict):
    predict = str(predict)
    if(int(predict) == 10):
        return 'A'
    elif(int(predict) == 11):
        return 'B'
    elif(int(predict) == 12):
        return 'C'
    elif(int(predict) == 13):
        return 'D'
    elif(int(predict) == 14):
        return 'E'
    elif(int(predict) == 15):
        return 'F'
    else:
        return predict


for i in range(50):
    img_predict = np.array(['', '', '', '', '', ''])
    name_test = './DATA_TEST/' + str(i) + ".png"
    img_test = cv2.imread(name_test)
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    img_test = cv2.resize(src=img_test, dsize=(1200, 400))
    ret, thresh = cv2.threshold(img_test, 127, 255, 0)
    print(type(img_test))
    coutours, hierarphy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    MAX_S = np.array([0, 1, 2, 3, 4, 5])
    coutours = np.asarray(coutours)
    crop_xywh = max_6(coutours)
    place = 15
    for j in range(6):
        crop_img = img_test[crop_xywh[j][1] - place:crop_xywh[j][1] + crop_xywh[j]
                            [3] + place, crop_xywh[j][0] - place:crop_xywh[j][0]+crop_xywh[j][2]+place]
        crop_img = cv2.resize(src=crop_img, dsize=(28, 28))

        y_predict = model.predict(crop_img.reshape(1, 28, 28, 1))
        img_predict[j] = convert_back(np.argmax(y_predict))
    print("Ket Qua La:", img_predict)
    rs = ""
    for i in img_predict:
        rs += i
    cv2.imwrite('./Results/%s.png' % rs, img_test)
