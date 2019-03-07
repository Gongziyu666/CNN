# -*- coding: utf-8 -*-
# @Time   : 2019/3/7 16:51
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: BaseModel.py
# @Software: PyCharm

import logging
from keras import Sequential
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense, Embedding
from keras.preprocessing.text import Tokenizer

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


class BaseModel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def model(self, x, y):
        model = Sequential()
        model.add(Conv1D(Embedding))
        model.add(MaxPool1D(2))
        model.add(Conv1D(filters=16, kernel_size=(3, 200), padding='same', activation='relu'))
        model.add(MaxPool1D(2))
        model.add(Dropout(rate=0.25))

        model.add(Conv1D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool1D(pool_size=(2, 2)))
        model.add(Conv1D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool1D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))

        model.add(Conv1D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool1D(pool_size=(2, 2), strides=2))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x, y, batch_size=64, epochs=20, validation_split=0.2, shuffle=True)
        return model

    def transform(self):
        tokenizer = Tokenizer(num_words=3000)
        tokenizer.fit_on_texts()

    def read_file(self,):
        pass
