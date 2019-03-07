# -*- coding: utf-8 -*-
# @Time   : 2019/3/6 17:30
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: 特征选择.py
# @Software: PyCharm

import logging
import matplotlib as mpl
import pandas as pd
from utils.seg import PkusegUtils
import numpy as np
from utils.string import category_to_id
mpl.rcParams['figure.figsize'] = (30, 20)
import matplotlib.pyplot as plt


base_dir = 'E:\\paper\\data\\'

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def open_file(file):
    data = []
    with open(base_dir + file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.replace('\n', '').split('\t'))
    return data


pks = PkusegUtils()
train = pd.DataFrame(open_file('train'), columns=['label', 'content'])
test = pd.DataFrame(open_file('test'), columns=['label', 'content'])
train['label'] = train['label'].apply(lambda x: category_to_id(x)).astype('float32')
# train['content'] = train['content'].apply(lambda x: pks.seg(x))
test['label'] = test['label'].apply(lambda x: category_to_id(x)).astype('float32')
# test['content'] = test['content'].apply(lambda x: pks.seg(x))

x_train = train.iloc[:, 1]
y_train = train.iloc[:, 0]
x_test = test.iloc[:, 1]
y_test = test.iloc[:, 0]











