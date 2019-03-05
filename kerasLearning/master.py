# -*- coding: utf-8 -*-
# @Time   : 2019/3/4 16:54
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: master.py
# @Software: PyCharm

# 导入相关库
import logging
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, Flatten

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 序列模型实现
model = Sequential()
model.add(Dense(units=32, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# 通用模型实现
X_input = Input(shape=(784, ))
dense_1 = Dense(units=32)(X_input)
act_1 = Activation('relu')(dense_1)
output = Dense(units=10, activation='softmax')(act_1)
model = Model(inputs=X_input, output=output)

