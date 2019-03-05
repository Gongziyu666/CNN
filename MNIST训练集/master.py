# -*- coding: utf-8 -*-
# @Time   : 2019/3/5 8:56
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: master.py
# @Software: PyCharm

import logging
from keras.datasets import mnist
from keras.utils import np_utils
from keras import Sequential, models
from keras.layers import Dense, Conv2D, MaxPool2D, Conv1D, Flatten, Dropout ,Lambda
import keras
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def LeNet5():
    model = Sequential()
    model.add(Lambda(lambda x: keras.backend.expand_dims(x)))
    model.add(Conv2D(filters=6, kernel_size=(5, 5),
                     padding='valid', activation='tanh',
                     input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                     activation='tanh', ))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dropout(rate=0.5))
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    return model


def train_model():
    model = LeNet5()

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1,
              validation_split=0.2, shuffle=True)
    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

x_train = x_train/255
x_test = x_test/255
y_train = np_utils.to_categorical(y=y_train, num_classes=10)
y_test = np_utils.to_categorical(y=y_test, num_classes=10)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
model = train_model()
loss, accuracy = model.evaluate(x_test, y_test)
print('loss = %f  accuracy = %f'%(loss, accuracy))
model.save('lenet5.h5')

