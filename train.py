import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import glob
import string
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

folder = ['1', '2', '3', '4', '5', '6', '7',
          '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']


def convertChar2Int(Y_TV):
    for i, Y in enumerate(Y_TV):
        if(Y == 'A'):
            Y = 10
        if(Y == 'B'):
            Y = 11
        if(Y == 'C'):
            Y = 12
        if(Y == 'D'):
            Y = 13
        if(Y == 'E'):
            Y = 14
        if(Y == 'F'):
            Y = 15
        Y_TV[i] = int(Y)
    return(Y_TV)


# =================GET DATA 1 -> 'F'==============================
Y_train = np.array([])
X_train = np.full((2912, 28, 28), 1)
print('Loading Data...')
for i in range(2912):
    Y_train = np.append(Y_train, '', axis=None)
print('Get Data...')
y = 0
for i in folder:
    for img in glob.glob("./DATA_TRAIN/" + i + '/*.png'):
        print('folder:', i, ' y ->', y)
        n = cv2.imread(img)
        n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        n = cv2.resize(src=n, dsize=(28, 28))
        X_train[y] = n
        Y_train[y] = i
        y += 1
print('-----Get Data Successfully-----')
X_val, Y_val = X_train[2812:2912, :], Y_train[2812:2912]  # numpy
X_train, Y_train = X_train[:2812, :], Y_train[:2812]  # numpy
# =======================Shape data==========================

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
# =======================One hot=============================
Y_train = convertChar2Int(Y_train)
Y_val = convertChar2Int(Y_val)
Y_train = np_utils.to_categorical(Y_train, 16)
Y_val = np_utils.to_categorical(Y_val, 16)
print('Y train:', Y_train.shape)
print('X train:', X_train.shape)
print('Y val:', Y_val.shape)
print('X val:', X_val.shape)
# 5. Định nghĩa model
model = Sequential()

# Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3
# dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))

# Thêm Convolutional layer
model.add(Conv2D(32, (3, 3), activation='sigmoid'))

# Thêm Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer chuyển từ tensor sang vector
model.add(Flatten())

# Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid
model.add(Dense(128, activation='sigmoid'))

# Output layer với 10 node và dùng softmax function để chuyển sang xác xuất.
model.add(Dense(16, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              batch_size=74, epochs=10, verbose=1)
# 8. Vẽ đồ thị loss, accuracy của traning set và validation set
fig = plt.figure()
numOfEpoch = 10

model.save('model.ckpt')
