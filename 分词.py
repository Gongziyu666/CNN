# -*- coding: utf-8 -*-
# @Time   : 2019/3/1 13:42
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: 分词.py
# @Software: PyCharm

# import logging

# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
from utils.seg import PkusegUtils, JiebaUtils


def read_stopwords(file):
    stop_words = []
    with open(file, 'r', encoding='utf-8') as f:
        for word in f.readlines():
            stop_words.append(word.replace('\n', ''))
    return stop_words


def save(l, path):
    with open(path, 'w+') as f:
        for words in l:
            f.write(words+'\n')


file = r'E:\\论文测试训练集\\data\\test.txt'
save_dir = r'E:\\论文测试训练集\\data\\'
pku = PkusegUtils()
jbu = JiebaUtils()
jb = []
ps = []
with open(file, 'r', encoding='utf-8') as f:
    stop_words = read_stopwords('stopwords.dic')
    for line in f.readlines():
        jb.append(' '.join([word for word in jbu.seg(line) if word not in stop_words]))
        ps.append(' '.join([word for word in pku.seg(line) if word not in stop_words]))

save(jb, save_dir+"jieba_remove_stopwords.txt")
save(ps, save_dir+"pkuseg_remove_stopwords.txt")






