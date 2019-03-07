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
from utils.string import category_to_id
from utils.seg import JiebaUtils
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


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

def read_file(file):
    path = 'E:\\paper\\data\\'
    labels = []
    contents = []
    jb = JiebaUtils()
    with open(path+file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.replace('\n', '').split('\t')
            categoryID = category_to_id(words[0])
            labels.append(categoryID)
            words = ' '.join(jb.seg(words[1]))
            contents.append(words)
    return contents, labels


if __name__ == '__main__':
    x_train, y_train = read_file('train')
    x_test, y_test = read_file('test')
    tokenizer = Tokenizer(num_words=3000)
    tokenizer.fit_on_texts(x_train)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train_seq, maxlen=200)
    x_test = pad_sequences(x_test_seq, maxlen=200)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(y_train[0])
    print(x_train[0])



