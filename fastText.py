from gensim.models import FastText
from gensim.test.utils import datapath
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


filePath = 'E:\\论文测试训练集\\corpusSegDone.txt'
saveFilePath = 'E:\\论文测试训练集\\fasttext_s200_w6_pkuseg_MODEL.bin'

corpus = datapath(filePath)
# w2vModel = word2vec.Word2Vec(sentences, size=200, window=5, min_count=5, iter=5,
#                                  workers=multiprocessing.cpu_count())
model = FastText(size=200, window=6, min_count=5, iter=30,
                 workers=multiprocessing.cpu_count())
model.build_vocab(corpus_file=corpus)
model.save(saveFilePath)
