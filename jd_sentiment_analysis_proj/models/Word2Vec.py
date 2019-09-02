# 训练word2vec词向量
import os
import time
from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences
import logging
import multiprocessing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class word2vec_config(object):
    def __init__(self,
                 embedding_size=300,  # 词向量的维度
                 win_size=5,   # 在一个句子中，当前词和预测词的最大距离(词向量上下文最大距离)
                 min_count=5,  # 词频少于min_count次数的单词会被丢弃掉
                 sg=0,  # 训练算法：sg=0 使用cbow训练, sg=1 使用skip-gram 对低频词较为敏感
                 n_iter=10,  # 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
                 cpu_count=multiprocessing.cpu_count()  # 设置多线程训练模型，机器的核数越多，训练越快
                 ):
        self.__embedding_size = embedding_size
        self.__win_size = win_size
        self.__min_count = min_count
        self.__sg = sg
        self.__n_iter = n_iter
        self.__cpu_count = cpu_count

    @property
    def embedding_size(self):
        return self.__embedding_size

    @property
    def win_size(self):
        return self.__win_size

    @property
    def min_count(self):
        return self.__min_count

    @property
    def sg(self):
        return self.__sg

    @property
    def n_iter(self):
        return self.__n_iter

    @property
    def cpu_count(self):
        return self.__cpu_count


'''
由于语料太大，不能一次性加载到内存训练，
gensim提供了PathLineSentences(input_dir)这个类，
会去指定目录依次读取语料数据文件，
采用iterator方式加载训练数据到内存
'''
def train_wd2vec(model_save_path, corpus_dir, config):
    t1 = time.time()
    word2vec_model = Word2Vec(PathLineSentences(corpus_dir),
                              size=config.embedding_size,
                              window=config.win_size,
                              min_count=config.min_count,
                              sg=config.sg,
                              workers=config.cpu_count,
                              iter=config.n_iter
                              )
    '''
    训练方式2：
        word2vec_model = Word2Vec(size=EMBEDDING_SIZE,
                                  window=WIN_SIZE,
                                  min_count=MIN_COUNT,
                                  sg=0,
                                  workers=CPU_COUNT,
                                  iter=N_ITER)
        word2vec_model.build_vocab(sentences)
        word2vec_model.train(sentences, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)        
    '''

    t2 = time.time()
    logging.info('词向量训练结束！总用时：{}min'.format((t2 - t1) / 60.0))

    word2vec_model.save(model_save_path)  # 保存词向量模型
    logging.info('词向量模型已保存......')


if __name__ == '__main__':
    model_save_path = 'word2vec.model'
    corpus_dir = 'data/word2vec_corpus/'
    config = word2vec_config()
    train_wd2vec(model_save_path, corpus_dir, config)
