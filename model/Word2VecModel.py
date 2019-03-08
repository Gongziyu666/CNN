# -*- coding: utf-8 -*-
# @Time   : 2019/3/7 16:48
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: Word2VecModel.py
# @Software: PyCharm

import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from utils.seg import PkusegUtils
from utils.string import category_to_id


def do_train(x, y, history):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(200, 200)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, batch_size=64, epochs=3, validation_split=0.2, shuffle=True, callbacks=[history])


def read_file(file):
    path = 'E:\\paper\\data\\'
    labels = []
    contents = []
    pku = PkusegUtils()
    with open(path+file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.replace('\n', '').split('\t')
            categoryID = category_to_id(words[0])
            labels.append(categoryID)
            words = ' '.join(pku.seg(words[1]))
            contents.append(words)
    return contents, labels





