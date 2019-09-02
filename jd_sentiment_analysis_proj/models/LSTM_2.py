'''
说明：
所有极性的评论分词都保存到文件中！
比较节省内存，但因文件读写，训练速度较慢！！！
'''

import os
import html
import time
import re
import csv
import numpy as np
import pickle
import yaml
import jieba
import jieba.analyse
import warnings
# import logging
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPool1D
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_yaml
from keras import backend as K
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec, PathLineSentences
from gensim.corpora.dictionary import Dictionary
from log.logger import MyLogger


#设置日志级别，默认是logging.WARNING，低于该级别的不会输出，级别排序:CRITICAL>ERROR>WARNING>INFO>DEBUG
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# logging.basicConfig(filename='log/new.log', #日志文件
#                     filemode='a', #模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志，a是追加模式，默认是追加模式
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', #日志格式
#                     level=logging.DEBUG,#控制台打印的日志级别
#                 ) #只会保存log到文件，不会输出到控制台

#随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数
np.random.seed(3347)
LABEL_POS, LABEL_NEG = 1, 0 #正样例标签1、负样例标签0
N_ITER = 8 #词向量训练的迭代次数
WIN_SIZE = 8  #词向量上下文最大距离，一般取[5-10]
MIN_COUNT = 10 #需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词
EMBEDDING_SIZE = 300 #词向量的维度，数据集越大该值越大
HIDDEN_LAYER_SIZE = 64 #隐层的大小
BATCH_SIZE = 64 #批的大小
NUM_EPOCHS = 10 #LSTM样本训练的轮数
TEST_SIZE = 0.2 #总样本中，测试样本的占比
CPU_COUNT = multiprocessing.cpu_count() #cpu线程数量

COMMENT_DIR = 'comment' #源评论文件目录
SEGMENT_DIR = 'segment' #分词文件目录
USER_DIR = 'user' #用户自定义文件目录
MODEL_DIR = 'model' #保存模型目录
USER_DICT_PATH = os.path.join(USER_DIR, 'user_dict.txt') #自定义用户表
STOP_WORDS_PATH = os.path.join(USER_DIR, 'stop_words.txt') #停用词表
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm.yml') #保存LSTM模型结构的路径
LSTM_WEIGHT_PATH = os.path.join(MODEL_DIR, 'lstm.h5') #保存LSTM权重的路径
# WORD2VEC_PATH = os.path.join(MODEL_DIR, 'word2vec_model.pkl') #保存word2vec词向量模型的路径
WORD2VEC_PATH = os.path.join(MODEL_DIR, 'word2vec.model') #保存word2vec词向量模型的路径
HIST_PATH = os.path.join(MODEL_DIR, 'hist.pkl') #保存训练过程中的历史记录
jieba.load_userdict(USER_DICT_PATH)
jieba.analyse.set_stop_words(STOP_WORDS_PATH)
pos_tags = ['n', 'vn', 'v', 'ad', 'a', 'e', 'y'] #是名词、形容词、动词、副词、叹词、语气词

'''
Word2vec对象还支持online learning。我们可以将更多的训练数据传递给一个已经训练好的word2vec对象，继续更新模型的参数：
model = gensim.models.Word2Vec.load('/tmp/mymodel')
model.train(more_sentences)
'''


#自定义文本处理工具类
class TextUtils:
    __min_len = 8  #评论文本长度至少为8

    # 加载停用词表
    @classmethod
    def stop_words(cls):
        with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
            stpwrds = f.read()  # 读取文件中所有内容
            stpwrdlst = stpwrds.splitlines()  # 按照行分隔，返回一个包含各行作为元素的列表，默认不包含换行符
            return stpwrdlst

    @classmethod
    def is_blank(cls, _str):
        if _str in ['', np.nan, None]:
            return True
        else:
            return False

    @classmethod
    def no_blank(cls, _str):  # 去除空白字符
        return re.sub('\s', '', _str)

    @classmethod
    def html_unescape(cls, _str):  # 将字符串中的html实体转换成html标签
        return html.unescape(_str)

    @classmethod
    def maketrans(cls, _str, src, target):  # 将源串中需要转换的字符（串）转换成目标字符（串）
        if src in _str and len(src) == len(target):
            trans_table = str.maketrans(src, target)  # 如：贼贵 -> 很贵
            return _str.translate(trans_table)
        else:
            return _str

    @classmethod
    def tokenize(cls, _str): #对评论列表进行分词
        return [wd for wd in jieba.lcut(re.sub('\s', '', _str)) if wd not in cls.stop_words()]

    @classmethod
    def del_repeat_elem_from_list(cls, _list):  # 删除列表中相邻重复元素，如：[1,2,2,3,3,3,4,4,5]->[1,2,3,4,5]
        result = []
        for item in _list:
            size = len(result)
            if size == 0 or item == 0 or result[size - 1] != item:
                result.append(item)
        return result

    @classmethod
    def del_repeat_chars_from_str(cls, _str):  # 删除字符串中连续重复的字符，如：abccdssbb -> abcdsb
        n = len(_str)  # 字符长度
        if n == 1:
            return _str
        _list = list(_str)  # 字符串列表化
        list1 = []
        for i in range(n - 1):
            if _list[i] != _list[i + 1]:
                list1.append(_list[i])
        list1.append(_list[-1])  # 添加末尾字符
        str1 = ''.join(list1)
        return str1

    @classmethod
    def del_repeat_words_from_str(cls, _str):  # 连续重复词语（压缩去重） acabababcdsab -> acabcdsab
        n = len(_str)
        if n <= 1:
            return _str
        rm_list = []
        i = 0
        idx = 0
        while i < n:
            flag = False
            for j in range(n - 1, i, -1):
                if j + j - i <= n:
                    if _str[i: j] == _str[j: (j + j - i)]:
                        rm_list.append([i, j])  # 保存重复序列的前后索引
                        idx = j
                        flag = True
                        break
            if flag:
                i = idx
            else:
                i += 1
        res = _str
        rm_len = 0
        for item in rm_list:
            res = res[:(item[0] - rm_len)] + res[(item[1] - rm_len):]
            rm_len += (item[1] - item[0])
        return res

    @classmethod
    def preprocess_sent(cls, sent):  #评论句子的预处理
        sent = cls.no_blank(sent)
        sent = cls.html_unescape(sent)
        sent = cls.del_repeat_words_from_str(sent)
        if cls.is_blank(sent) or len(sent) <= cls.__min_len: #忽略长度过短的评论（没有多大意义）
            return np.nan  #定义为空缺数据
        return sent


class SentimentModel:
    def __init__(self):
        self.max_len = 100  # 文本保留的最大长度

        self.logging = MyLogger()  # 自定义日志器
        self.margin = 0.6  # 阈值
        #self.theta = lambda t: (K.sign(t)+1.)/2.
        self.theta = lambda t: K.sigmoid(100 * t)  # 软化

        self.logging.info('###该模型为情感二分类模型###')
        self.logging.info('CPU线程数：{}'.format(CPU_COUNT))

    # 二分类：自定义损失函数（hinge loss+triplet loss）
    def loss_new(self, y_true, y_pred):
        return -(1 - self.theta(y_true - self.margin) * self.theta(y_pred - self.margin) - self.theta(
            1 - self.margin - y_true) * self.theta(1 - self.margin - y_pred)) * (
                           y_true * K.log(y_pred + K.epsilon()) + (1 - y_true) * K.log(1 - y_pred + K.epsilon()))

    # 二分类：focal_loss损失函数，解决样本不均衡分布问题
    def focal_loss_binary(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        pt_1 = tf.where(tf.equal(y_true, LABEL_POS), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, LABEL_NEG), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1. - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    # 加载停用词表
    def load_stop_words(self):
        with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
            stpwrd_content = f.read()  # 读取文件中所有内容
            stpwrdlst = stpwrd_content.splitlines() # 按照行分隔，返回一个包含各行作为元素的列表，默认不包含换行符
            return stpwrdlst

    '''
    input_csv_dir：csv格式的评论数据集目录
    output_txt_dir：分词后的txt文件目录（空格隔开）
    返回值：样本标签
    '''
    def corpus_seg(self, input_csv_dir, output_txt_dir):
        maxlen = 0
        labels = []
        if not os.path.exists(output_txt_dir):
            os.mkdir(output_txt_dir)

        for f in os.listdir(input_csv_dir):
            sent_size = 0
            csv_file = os.path.join(input_csv_dir, f)
            with open(csv_file, 'r') as fin:
                reader = csv.DictReader(fin)
                output_file = os.path.join(output_txt_dir, f[:f.rindex('.')] + '_seg.txt')
                with open(output_file, 'a', encoding='utf-8') as fout:
                    for row in reader:
                        comment = TextUtils.preprocess_sent(row['comment']) # comment是表第一行的某个数据，作为key
                        if not TextUtils.is_blank(comment):
                            words = jieba.lcut(comment)
                            if len(words) > maxlen:
                                maxlen = len(words)
                            segs = ' '.join(words)
                            fout.write('{}\n'.format(segs))
                            sent_size += 1
            if 'pos' in f:
                labels.extend([LABEL_POS] * sent_size)
            elif 'neg' in f:
                labels.extend([LABEL_NEG] * sent_size)

        self.max_len = maxlen
        self.logging.info('数据集总记录数：{}'.format(len(labels)))
        n_pos, n_neg = labels.count(LABEL_POS), labels.count(LABEL_NEG)
        self.logging.info('正样例数：{} 负样例数：{}'.format(n_pos, n_neg))
        return np.array(labels, dtype=int)


    #训练词向量模型
    #由于语料太大，不能一次性加载到内存训练，gensim提供了PathLineSentences(input_dir)这个类，会去指定目录依次读取语料数据文件，采用iterator方式加载训练数据到内存
    def train_wd2vect(self, seg_dir):  # 保存分词（空格隔开）的文件夹路径
        self.logging.info('开始训练词向量模型......')
        t1 = time.time()
        word2vec_model = Word2Vec(PathLineSentences(seg_dir),
                                  size=EMBEDDING_SIZE, #词向量的维度
                                  window=WIN_SIZE,  #在一个句子中，当前词和预测词的最大距离(词向量上下文最大距离)
                                  min_count=MIN_COUNT, #词频少于min_count次数的单词会被丢弃掉
                                  sg=0, #训练算法：sg=0 使用cbow训练, sg=1 使用skip-gram 对低频词较为敏感
                                  workers=CPU_COUNT, #设置多线程训练模型，机器的核数越多，训练越快
                                  iter=N_ITER  #随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
                                )
        t2 = time.time()
        self.logging.info('词向量训练结束！总用时：{}min'.format((t2 - t1) / 60.0))

        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)
        word2vec_model.save(WORD2VEC_PATH) #保存词向量模型
        self.logging.info('词向量模型已保存......')
        # word2vec_model.save_word2vec_format(out.model, binary=False)
        return word2vec_model


    # 根据词向量模型得到词索引{词: 索引} 和 词向量{词: 词向量}
    def create_dicts(self, wd2vec_model):
        if wd2vec_model is not None:
            gensim_dict = Dictionary()  # {索引: 词}
            # 实现词袋模型
            gensim_dict.doc2bow(wd2vec_model.wv.vocab.keys(), allow_update=True)  # (token_id, token_count)
            word2index = {wd: idx + 1 for idx, wd in gensim_dict.items()}  # 词索引字典 {词: 索引}，索引从1开始计数
            word_vectors = {wd: wd2vec_model.wv[wd] for wd in word2index.keys()}  # 词向量 {词: 词向量}
            return word2index, word_vectors
        else:
            return None


    # 获取字典长度和权重矩阵
    def get_embedding_weights(self, word2index, word_vectors):
        vocab_size = len(word2index) + 1  # 字段大小(索引数字的个数)，因为有的词语索引为0，所以+1
        embedding_weights = np.zeros((vocab_size, EMBEDDING_SIZE))  # vocab_size * EMBEDDING_SIZE的0矩阵
        for wd, idx in word2index.items():  # 从索引为1的词语开始，用词向量填充矩阵
            embedding_weights[idx, :] = word_vectors[wd]  # 词向量矩阵，第一行是0向量（没有索引为0的词语）
        return embedding_weights


    #构建CNN模型
    def build_cnn_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(GlobalMaxPool1D())
        model.add(Dense(1, activation="sigmoid"))
        # model.add(Dense(2, activation='softmax'))

        model.summary()
        return model



    #构建LSTM模型
    def build_lstm_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            mask_zero=True,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()
        return model


    #构建LSTM-CNN模型
    def build_lstm_cnn_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            mask_zero=True,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(GlobalMaxPool1D())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()
        return model


    #构建Bi-LSTM模型
    def build_bilstm_model(self, embedding_weights):
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=EMBEDDING_SIZE,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
                                merge_mode='concat'))
        # model.add(Bidirectional(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
        # model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        # inputs = Input(shape=(None,))
        # embedded = Embedding(vocab_size, EMBEDDING_SIZE, input_length=self.max_len)(inputs)
        # lstm_out = LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(embedded)
        # predict = Dense(1, activation='softmax')(lstm_out)
        # model = Model(inputs=inputs, outputs=predict)
        model.summary()
        return model


    #构建并训练LSTM模型
    def train_lstm_model(self, embedding_weights, X, y):
        lstm_model = self.build_lstm_model(embedding_weights)
        lstm_model.compile(loss=self.focal_loss, optimizer="adam", metrics=["accuracy"])
        es = EarlyStopping(monitor='val_loss', patience=NUM_EPOCHS/2)  # 经过NUM_EPOCHS/2轮训练，当val_acc不再提升，则停止训练
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
        self.logging.info('开始训练LSTM模型......')
        self.logging.info('训练集大小：{} 测试集大小：{}'.format(round((1-TEST_SIZE)*len(X)), round(len(X)*TEST_SIZE)))
        t1 = time.time()
        hist = lstm_model.fit(Xtrain,
                        ytrain,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=[es],
                        validation_data=(Xtest, ytest))
        t2 = time.time()
        self.logging.info('LSTM模型训练结束！总用时：{}min'.format((t2 - t1) / 60.0))

        with open(HIST_PATH, 'wb') as output: #保存history对象
            pickle.dump(hist.history, output)

        yaml_string = lstm_model.to_yaml()  # 保存模型结构为YAML字符串
        with open(LSTM_MODEL_PATH, 'w') as fout:
            fout.write(yaml.dump(yaml_string, default_flow_style=True))
        lstm_model.save_weights(LSTM_WEIGHT_PATH)  # 保存模型权重
        self.logging.info('LSTM模型已经保存......')

        self.logging.info('metrics names:'+lstm_model.metrics_names)
        loss, acc = lstm_model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
        self.logging.info('模型评估结果：loss:{} acc:{}'.format(loss, acc))


    def plot_hist(self):
        # plot loss and accuracy
        with open(HIST_PATH, 'rb') as pkl_hist:
            history = pickle.load(pkl_hist)
        plt.subplot(211)
        plt.title("Accuracy")
        plt.plot(history["acc"], color="g", label="Train")
        plt.plot(history["val_acc"], color="b", label="Validation")
        plt.legend(loc="best")
        plt.subplot(212)
        plt.title("Loss")
        plt.plot(history["loss"], color="g", label="Train")
        plt.plot(history["val_loss"], color="b", label="Validation")
        plt.legend(loc="best")
        plt.tight_layout() #自动调整子图间的间距
        plt.show()

    #加载LSTM模型
    def load_lstm_model(self):
        self.logging.info('loading models......')
        with open(LSTM_MODEL_PATH, 'r') as f:
            yaml_string = yaml.load(f)
        lstm_model = model_from_yaml(yaml_string)
        self.logging.info('loading weights......')
        lstm_model.load_weights(LSTM_WEIGHT_PATH)
        lstm_model.compile(loss=self.focal_loss, optimizer="adam", metrics=["accuracy"])
        return lstm_model


    #将分词序列转换成索引序列（分词序列由list列表传入）
    def text2index_from_lst(self, word2index, comm_seqs):
        data = []
        for seqs in comm_seqs:
            wd_idxs = []
            for wd in seqs:
                if wd in word2index.keys():
                    wd_idxs.append(word2index[wd])  # 单词转索引数字
                else:
                    wd_idxs.append(0)  # 索引字典里没有的词转为数字0
            data.append(TextUtils.del_repeat_elem_from_list(wd_idxs))

        return sequence.pad_sequences(data, self.max_len)  # 对齐序列


    # 将分词序列转换成索引序列（分词序列从文件中读取）
    def text2index_from_dir(self, word2index, seg_dir):
        data = []
        for file in os.listdir(seg_dir):
            f = os.path.join(seg_dir, file)
            with open(f, 'r', encoding='utf-8') as fin:
                for seqs in fin:
                    wd_idxs = []
                    for wd in seqs.split():
                        if wd in word2index.keys():
                            wd_idxs.append(word2index[wd])  # 单词转为索引
                        else:
                            wd_idxs.append(0)  # 索引字典里没有的词转为数字0
                    data.append(TextUtils.del_repeat_elem_from_list(wd_idxs))

        return sequence.pad_sequences(data, self.max_len)  # 按最大长度对齐序列


    #情感预测（传入评论列表）
    def predict_by_lst(self, new_comms):
        lstm_model = self.load_lstm_model() #LSTM模型
        word2vec_model = Word2Vec.load(WORD2VEC_PATH) #词向量模型
        word2index, _ = self.create_dicts(word2vec_model)
        new_comms = [TextUtils.preprocess_sent(com) for com in new_comms if not TextUtils.is_blank(com)]
        wd_seqs = [TextUtils.tokenize(com) for com in new_comms if not TextUtils.is_blank(com)]
        X = self.text2index_from_lst(word2index, wd_seqs)
        for i, x in enumerate(X):
            x = np.array(x).reshape(1, -1)
            res = lstm_model.predict_classes(x)
            if res[0][0] == LABEL_POS:
                print(new_comms[i], '\n', 'positive!')
            else:
                print(new_comms[i], '\n', 'negative!')


    #情感预测（传入文件路径）
    def predict_by_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as fin:
            # new_comms = fin.read().splitlines()
            new_coms = [line.strip() for line in fin if not TextUtils.is_blank(line)]
            self.predict_by_lst(new_coms)


    #训练
    def train(self):
        comment_dir = COMMENT_DIR
        seg_dir = SEGMENT_DIR
        # 加载数据
        labels = self.corpus_seg(comment_dir, seg_dir)
        # 训练词向量模型
        word2vec_model = self.train_wd2vect(seg_dir)
        # 创建索引词典和词向量
        word2index, word_vectors = self.create_dicts(word2vec_model)
        # 文本序列索引化
        comms_seqs = self.text2index_from_dir(word2index, seg_dir)
        # 获取权值矩阵
        embedding_weights = self.get_embedding_weights(word2index, word_vectors)
        # 序列LSTM
        self.train_lstm_model(embedding_weights, comms_seqs, labels)


if __name__ == "__main__":
    model = SentimentModel()
    model.train()

    new_comms = ['特别卡，新机都很卡，而且买完就掉价，当备用机可以，其他的不建议购买，包装很差，就一个袋子包着一个盒子',
                 '给父亲买的手机，功能全面，性价比很高',
                 '烂烂烂！服务烂，价格烂，功能也烂，没有一点好处，小米也就这样了，时日不多了！',
                 '用一个月的手机就冲不进电了，只能说现在的红米手机太垃圾了，大家别买了，便宜没好货',
                 '手机收到了，给妈妈买的。功能都很好。很实用。',
                 '买回来就不喜欢了，给老公买的，那会不在家，都没用过，又过了七天无理由退款了，丢在抽屉里呢',
                 '到货第二天就降了50元，前置摄像头像素差，后置的摄像头感觉特别突出手机平面。',
                 '电池不怎么好！充满电一次最多用12个小时，还得是不玩游戏，不看视频，也就打打电话，感觉不好。']
    model.predict_by_lst(new_comms)

    # model.predict_by_file("user/samples.txt")