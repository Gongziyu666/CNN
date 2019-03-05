# -*- coding: utf-8 -*-
# @Time   : 2019/3/5 10:46
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: test.py
# @Software: PyCharm

import logging
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.datasets import cifar10
from keras.utils import np_utils

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=64, epochs=20, validation_split=0.2, shuffle=True)
    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y=y_train, num_classes=10)
y_test = np_utils.to_categorical(y=y_test, num_classes=10)

print(x_train.shape, y_train.shape)

model = model()
loss, accuracy = model.evaluate(x_test, y_test)
print('loss = %f  ========> accuracy = %f' % (loss, accuracy))


