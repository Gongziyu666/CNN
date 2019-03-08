# -*- coding: utf-8 -*-
# @Time   : 2019/3/7 16:51
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: TokenizerModel.py
# @Software: PyCharm

import logging
from keras import Sequential
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense, Embedding, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from utils.string import category_to_id
from utils.seg import JiebaUtils
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model.CallBlack import LossHistory

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def do_train(x, y, history, maxlen=200, max_features=3000, embed_size=200):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.2))
    filter_size = [2, 3, 4, 5]
    for fsz in filter_size:
        model.add(Conv1D(filters=200, kernel_size=fsz, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, batch_size=64, epochs=3, validation_split=0.2, shuffle=True, callbacks=[history])
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
    history = LossHistory()
    model = do_train(x_train, y_train, history)
    model.save(r'E:\paper\data\tokenizerModel.model')
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss = %s ======> accuracy = %s' % (loss, accuracy))
    history.loss_plot('epoch')



