# -*- coding: utf-8 -*-
# @Time   : 2019/3/5 14:04
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: textsplit.py
# @Software: PyCharm

import logging
import os

from utils.seg import PkusegUtils

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

source_file = r'E:\论文测试训练集\data\train'


def open_file(file):
    x = []
    y = []
    with open(file, 'r', encoding='utf-8') as f:
        pk = PkusegUtils()
        for line in f.readlines():
            label, content = line.split("\t")
            y.append(label)
            x.append(pk.seg(content))

    return x, y


y_train, x_train = open_file(source_file)

size = [len(line) for line in x_train]



