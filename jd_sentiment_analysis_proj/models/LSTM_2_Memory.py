'''
说明：
没有文件写操作，完全将数据集加载到内存中进行操作
对于较大的数据集会很耗内存，但训练速度快！！！
'''

import os
import time
import numpy as np
import pandas as pd
import pickle
import yaml
import jieba.analyse
import warnings
# import logging
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
# from collections import Counter
import keras
from keras import losses
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D, Convolution1D
from keras.layers.pooling import GlobalMaxPool1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential, model_from_yaml
from keras import backend as K
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from log.logger import MyLogger
from models.text_utils import TextUtils

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
N_ROWS = 260000 #每个类别的评论数
N_ITER = 8  #词向量训练的迭代次数
WIN_SIZE = 8  #词向量上下文最大距离，一般取[5-10]
MIN_COUNT = 5 #需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词
EMBEDDING_SIZE = 300 #词向量的维度，数据集越大该值越大
HIDDEN_LAYER_SIZE = 64 #隐层的大小
BATCH_SIZE = 64 #批的大小
NUM_EPOCHS = 10 #LSTM样本训练的轮数
TEST_SIZE = 0.2 #总样本中，测试样本的占比
CPU_COUNT = multiprocessing.cpu_count() #cpu线程数量

COMMENT_DIR = '../comment' #源评论文件目录
USER_DIR = '../user' #用户自定义文件目录
MODEL_DIR = '../log' #保存模型目录
USER_DICT_PATH = os.path.join(USER_DIR, 'user_dict.txt') #自定义用户表
STOP_WORDS_PATH = os.path.join(USER_DIR, 'stop_words.txt') #停用词表
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm.yml') #保存LSTM模型结构的路径
LSTM_WEIGHT_PATH = os.path.join(MODEL_DIR, 'lstm.h5') #保存LSTM权重的路径
# WORD2VEC_PATH = os.path.join(MODEL_DIR, 'word2vec_model.pkl') #保存word2vec词向量模型的路径
WORD2VEC_PATH = os.path.join('../model_lstm', 'word2vec.model') #保存word2vec词向量模型的路径
HIST_PATH = os.path.join(MODEL_DIR, 'hist.pkl') #保存训练过程中的历史记录
jieba.load_userdict(USER_DICT_PATH)
jieba.analyse.set_stop_words(STOP_WORDS_PATH)
pos_tags = ['n', 'vn', 'v', 'ad', 'a', 'e', 'y'] #是名词、形容词、动词、副词、叹词、语气词
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'model-{epoch:02d}-{val_acc:.2f}.h5') #保存训练最好的模型的权值


class SentimentModel:
    def __init__(self):
        self.max_len = 100  # 文本保留的最大长度

        self.logging = MyLogger()  # 自定义日志器
        self.margin = 0.6  # 阈值
        self.theta = lambda t: (K.sign(t)+1)/2
        # self.theta = lambda t: K.sigmoid(100 * t)  # 软化
        self.logging.info('###该模型为情感二分类模型###')
        self.logging.info('CPU线程数：{}'.format(CPU_COUNT))

    # 二分类：自定义损失函数（hinge loss+triplet loss）
    def loss_new(self, y_true, y_pred):
        return -(1 - self.theta(y_true - self.margin) * self.theta(y_pred - self.margin) - self.theta(
            1 - self.margin - y_true) * self.theta(1 - self.margin - y_pred)) * (
                           y_true * K.log(y_pred + K.epsilon()) + (1 - y_true) * K.log(1 - y_pred + K.epsilon()))

    # 二分类：focal_loss损失函数，解决样本不均衡分布问题
    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        pt_1 = tf.where(tf.equal(y_true, LABEL_POS), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, LABEL_NEG), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1. - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    '''
     input_csv_dir：csv格式的评论数据集目录
     返回值：DataFrame对象
     '''
    def load_train_data(self, input_csv_dir):
        self.logging.info('开始加载数据......')
        t1 = time.time()
        df_data = pd.DataFrame()
        if os.path.isdir(input_csv_dir):
            for f in os.listdir(input_csv_dir):
                csv_file = os.path.join(input_csv_dir, f)
                csv_data = pd.read_csv(csv_file, usecols=['label', 'segs'], nrows=N_ROWS)
                df_data = df_data.append(csv_data, ignore_index=True)
        else:
            df_data = pd.read_csv(input_csv_dir, usecols=['label', 'segs'], nrows=N_ROWS)

        t2 = time.time()
        self.logging.info('数据加载完成！总用时：{}s'.format(t2 - t1))
        # self.max_len = max(df['segs'].apply(lambda x: len(x)))  # 序列的最大长度

        self.logging.info('数据集总记录数：{}'.format(len(df_data)))
        nb_pos = len(df_data[df_data['label'] == LABEL_POS])  # 正样例数
        nb_neg = len(df_data[df_data['label'] == LABEL_NEG])  # 负样例数
        self.logging.info('正面例数：{} 负面样例数：{}'.format(nb_pos, nb_neg))
        return shuffle(df_data)  # 随机打乱数据集

    #训练词向量模型
    def train_wd2vect(self, sentences): #分词列表
        if os.path.exists(WORD2VEC_PATH):
            word2vec_model = Word2Vec.load(WORD2VEC_PATH)
        else:
            self.logging.info('开始训练词向量模型......')
            t1 = time.time()
            word2vec_model = Word2Vec(sentences,
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
        model.add(Convolution1D(filters=128, kernel_size=3, activation='relu'))
        model.add(GlobalMaxPool1D())  # 对于时间信号的全局最大池化,MaxPooling1D限制每一步的池化大小
        model.add(Dense(1, activation="sigmoid"))
        # model.add(Dense(2, activation='softmax'))

        model.summary()
        return model

    # def build_cnn_model(self, embedding_weights):
    #     model = Sequential()
    #     vocab_size = len(embedding_weights)
    #     # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    #     model.add(Embedding(input_dim=vocab_size,  # 字典长度
    #                         output_dim=EMBEDDING_SIZE,
    #                         input_length=self.max_len,
    #                         weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
    #     model.add(SpatialDropout1D(0.2))
    #     model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    #     model.add(MaxPooling1D(pool_size=3)) #对于时间信号的全局最大池化,MaxPooling1D限制每一步的池化大小
    #     model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    #     model.add(MaxPooling1D(pool_size=3))  # 对于时间信号的全局最大池化,MaxPooling1D限制每一步的池化大小
    #     model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    #     model.add(Flatten())
    #     model.add(Dense(1, activation="sigmoid"))
    #     # model.add(Dense(2, activation='softmax'))
    #     model.summary()
    #     return model

    #构建LSTM模型
    def build_lstm_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            # mask_zero=True,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Dense(HIDDEN_LAYER_SIZE//2, activation='relu'))
        model.add(Dropout(0.3))
        # model.add(Dense(1, activation="sigmoid"))
        model.add(Dense(2, activation="softmax"))

        model.summary()
        return model

    #构建CNN-LSTM模型
    def build_cnn_lstm_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        # model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
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
                            # mask_zero=True,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
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
        model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2)))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        # inputs = Input(shape=(None,))
        # embedded = Embedding(vocab_size, EMBEDDING_SIZE, input_length=self.max_len)(inputs)
        # lstm_out = LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(embedded)
        # predict = Dense(1, activation='softmax')(lstm_out)
        # model = Model(inputs=inputs, outputs=predict)
        model.summary()
        return model

    def F1_score(self, y_true, y_pred):
        # Only computes a batch-wise average of recall.
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        # Only computes a batch-wise average of precision.
        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    #构建并训练LSTM模型
    def train_lstm_model(self, embedding_weights, X, y):
        lstm_model = self.build_lstm_model(embedding_weights)
        # lstm_model.compile(loss=losses.binary_crossentropy, optimizer="adam", metrics=["acc", self.F1_score])
        lstm_model.compile(loss=self.loss_new, optimizer="adam", metrics=["acc", self.F1_score])
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=True)
        ytrain = keras.utils.to_categorical(ytrain, num_classes=2)
        ytest = keras.utils.to_categorical(ytest, num_classes=2)

        self.logging.info('开始训练LSTM模型......')
        self.logging.info('训练集大小：{} 测试集大小：{}'.format(round((1-TEST_SIZE)*len(X)), round(len(X)*TEST_SIZE)))
        t1 = time.time()
        # 保存训练最好的模型（每3轮训练检测一次）
        # cp = ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_acc', mode='max', save_best_only=True, verbose=1,
        #                      period=2)
        # tb = TensorBoard(log_dir='log')
        es = EarlyStopping(monitor='val_loss', patience=NUM_EPOCHS / 2)  # 经过NUM_EPOCHS/2轮训练，当val_acc不再提升，则停止训练
        callbacks_lst = [es]
        hist = lstm_model.fit(Xtrain,
                        ytrain,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=callbacks_lst,
                        validation_data=(Xtest, ytest))
        t2 = time.time()
        self.logging.info('LSTM模型训练结束！总用时：{}min'.format((t2 - t1) / 60.0))

        with open(HIST_PATH, 'wb') as output: #保存history对象
            pickle.dump(hist.history, output)

        # yaml_string = lstm_model.to_yaml()  # 保存模型结构为YAML字符串
        # with open(LSTM_MODEL_PATH, 'w') as fout:
        #     fout.write(yaml.dump(yaml_string, default_flow_style=True))
        # lstm_model.save_weights(LSTM_WEIGHT_PATH)  # 保存模型权重
        # self.logging.info('LSTM模型已经保存......')

        self.logging.info(lstm_model.metrics_names)
        loss, acc, F1_score = lstm_model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE//2)
        self.logging.info('模型评估结果：loss:{} acc:{} F1:{}'.format(loss, acc, F1_score))

    def plot_hist(self):
        # plot loss and accuracy
        with open(HIST_PATH, 'rb') as pkl_hist:
            history = pickle.load(pkl_hist)
        plt.subplot(211)
        plt.title("Accuracy")
        # plt.plot(history["acc"], color="g", label="Train")
        plt.plot(history["val_acc"], color="b", label="Validation")
        plt.legend(loc="best")
        plt.subplot(212)
        plt.title("Loss")
        # plt.plot(history["loss"], color="g", label="Train")
        plt.plot(history["val_loss"], color="r", label="Validation")
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
            data.append(wd_idxs)

        return sequence.pad_sequences(data, self.max_len)  # 对齐序列



    #情感预测（传入评论列表）
    def predict_by_lst(self, new_comms):
        lstm_model = self.load_lstm_model() #LSTM模型
        word2vec_model = Word2Vec.load(WORD2VEC_PATH) #词向量模型
        word2index, _ = self.create_dicts(word2vec_model)
        new_comms = [TextUtils.process(com) for com in new_comms if not TextUtils.is_blank(com)]
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
    def train(self, comment_dir=COMMENT_DIR):
        #加载数据
        df_data = self.load_train_data(comment_dir)
        #训练词向量模型
        word2vec_model = self.train_wd2vect(df_data['segs'])
        #创建索引词典和词向量
        word2index, word_vectors = self.create_dicts(word2vec_model)
        # 获取权值矩阵
        embedding_weights = self.get_embedding_weights(word2index, word_vectors)
        #文本序列索引化
        comms_seqs = self.text2index_from_lst(word2index, df_data['segs'])
        #序列LSTM
        self.train_lstm_model(embedding_weights, comms_seqs, df_data['label'])


if __name__ == "__main__":
    model = SentimentModel()
    model.train('data/test.csv')
    model.plot_hist()
    # new_comms = ['性价比非常高，显示屏幕高清，处理速度快！不带鼠标。很轻，没有想象中的薄。自带的office365需要激活',
    #              '非常好看！很满意，这个价钱买到质量这么好的鞋子简直太感动了，客服推荐的码数刚好，很合脚，穿起来不磨脚，防滑，耐脏，绝对值！',
    #              '什么破商家。我申请换货，不让我换，硬要求我退货，我说不退，直接把我电话挂掉，服务差评',
    #              '冰箱收到了，空间挺大的，冷冻室有点小，能效是三级，冷冻，冷藏，保鲜三室可以调温，送来的时候冰箱里面有点味道！送电以后就没了！总得来说这次购物不错！',
    #              '发申通物流，很不靠谱，太慢了，本来就是冲着京东物流才买的，结果发的申通，下次再不买它家的了',
    #              '鞋子已是第二双，朋友看到好看帮忙买的，质量杠杠的，又轻又舒服，春天到了配啥衣服都好看，出去玩也不怕累脚了，值得推荐！！！！',
    #              '用了两天才过来评论的，首先电脑颜值高，四面窄观影简直不要太舒服，背光键盘也很nice，尺寸大小也很合适，开机使用反应超快的。刚拿到手就插上电@了，语音助手指导小白也能上手。值得购买！',
    #              '很差，给老爸买的靴子，本来想着尽一份孝心，结果给我的感触很大，家里在县城，那么多快递可以到，偏偏发一个到不了的快递，也不跟买家核实地址，也不打招呼，哪怕跟我们核实一下，我们可以自己选择快递啊。那么大老远，还下着大雨，直接让老人去取。这次购物体验很差，很差。卖家服务态度也很差，很差']
    # model.predict_by_lst(new_comms)

    # model.predict_by_file("user/samples.txt")