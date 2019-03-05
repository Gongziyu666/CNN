# -*- coding: utf-8 -*-
# @Time   : 2019/3/5 13:52
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: master.py
# @Software: PyCharm

import logging
from keras.preprocessing import sequence
from gensim.models import FastText

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

fastModel = FastText.load(r'E:\论文测试训练集\fasttext_s200_w6_pkuseg_MODEL.bin')
print(fastModel.wv['我爱中国'])