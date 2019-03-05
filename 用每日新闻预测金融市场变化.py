import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import date

# 读入数据
data = pd.read_csv('Combined_News_DJIA.csv')

# 分割数据
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] < '2014-12-31']

# 数据集处理
X_train = train[train.columns[2:]]
#去掉日期和标签，把每条新闻做成单独的句子，集合在一起
corpus = X_train.values.flatten().astype(str)
#获取语料库
X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])

X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values
y_test = test['Label'].values

# 分词
from nltk.tokenize import word_tokenize
import nltk
nltk.download()
corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]


# 预处理
# 停止词
from nltk.corpus import stopwords
stop = stopwords.words('english')

# 数字
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
# 正则表达式，出现数字的时候返回true

# 特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))
"""
正则表达式，\w表示单词字符[A-Za-z0-9_]
[^\w]表示取反==\W非单词字符
全是特殊符号时返回true
"""

# lemma
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word= word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

# 把上面的方法综合起来
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)