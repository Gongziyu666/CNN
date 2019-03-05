# -*- coding: utf-8 -*-
# @Time   : 2019/3/4 14:12
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: master.py
# @Software: PyCharm

import logging
import os

from keras import Input, Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import metrics

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def read_files(filetype):
    """
    :param filetype: 'train' or test
    :return:
    all_texts:filetype 数据集文本
    all_labels:filetype 数据集标签
    """
    all_labels = [1]*12500+[0]*12500
    all_texts = []
    file_list = []
    root_dir = r'E:/aclImdb/'

    pos_path = root_dir+filetype+'\\pos\\'
    neg_path = root_dir+filetype+'\\neg\\'

    for file in os.listdir(pos_path):
        file_list.append(pos_path+file)
    for file in os.listdir(neg_path):
        file_list.append(neg_path+file)
    for file_name in file_list:
        with open(file_name, encoding='utf-8') as f:
            all_texts.append(' '.join(f.readlines()))
    return all_texts, all_labels


def text_cnn(maxlen= 200, max_features= 3000, embed_size= 32):
    comment_seq = Input(shape=[maxlen] , name='x_seq')
    emb_comment = Embedding(max_features, embed_size)(comment_seq)
    convs = []
    filter_size = [2, 3, 4, 5]
    for fsz in filter_size:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='tanh')(emb_comment)
        l_pool = MaxPooling1D(maxlen-fsz+1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=1, activation='sigmoid')(output)

    model = Model([comment_seq], output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_text, y_train = read_files('train')
    test_text, y_test = read_files('test')
    tokenizer = Tokenizer(num_words=3000)

    tokenizer.fit_on_texts(train_text + test_text)

    X_train_seq = tokenizer.texts_to_sequences(train_text)
    X_test_seq = tokenizer.texts_to_sequences(test_text)
    X_train = sequence.pad_sequences(X_train_seq, maxlen=200)
    X_test = sequence.pad_sequences(X_test_seq, maxlen=200)
    model = text_cnn()
    batch_size = 64
    epochs = 10
    model.fit(X_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    scores = model.evaluate(X_test, y_test)

    print('test_loss:%f, accuracy:%f' % (scores[0], scores[1]))


