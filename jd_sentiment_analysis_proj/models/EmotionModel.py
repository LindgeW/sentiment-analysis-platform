'''
说明：
没有文件写操作，完全将数据集加载到内存中进行操作
对于较大的数据集会很耗内存，但训练速度快！！！
'''
import os
import numpy as np
import yaml
import keras
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import tensorflow as tf
from keras import losses
from keras.preprocessing import sequence
from keras.models import model_from_yaml
from keras import backend as K
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from models.text_utils import TextUtils
keras.backend.clear_session()

# 设置日志级别，默认是logging.WARNING，低于该级别的不会输出，级别排序:CRITICAL>ERROR>WARNING>INFO>DEBUG
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# logging.basicConfig(filename='log/new.log', #日志文件
#                     filemode='a', #模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志，a是追加模式，默认是追加模式
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', #日志格式
#                     level=logging.DEBUG,#控制台打印的日志级别
#                 ) #只会保存log到文件，不会输出到控制台

# 随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数
np.random.seed(3347)

MODEL_DIR = '../model_3_lstm'  # 保存模型目录
# MODEL_DIR = '../models/model'  # 保存模型目录
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm.yml')  # 保存LSTM模型结构的路径
LSTM_WEIGHT_PATH = os.path.join(MODEL_DIR, 'model-08-0.91.h5')  # 保存LSTM权重的路径
# LSTM_WEIGHT_PATH = os.path.join(MODEL_DIR, 'model-04-0.92.h5')  # 保存LSTM权重的路径
WORD2VEC_PATH = os.path.join(MODEL_DIR, 'word2vec.model')  # 保存word2vec词向量模型的路径


class SentimentModel:
    def __init__(self):
        self.max_len = 100  # 文本保留的最大长度
        self.lstm_model = None
        self.word2index = None

    def load(self):
        # assert os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_WEIGHT_PATH)
        self.lstm_model = self.load_lstm_model(model_path=LSTM_MODEL_PATH, weight_path=LSTM_WEIGHT_PATH)  # LSTM模型
        print(self.lstm_model.summary())
        self.word2index = self.get_word_indexs(Word2Vec.load(WORD2VEC_PATH))

    # 多分类：训练的本质就是寻找损失函数最小值的过程，y_true为网络给出的预测值，y_true即是标签，两者均为tensor
    def focal_loss_multi_class(self, y_true, y_pred, e=0.1, alpha=0.25, gamma=2):
        # 公式：fl = -alpha * (1-pt)^gamma*log(pt)
        ce = - y_true * K.log(y_pred + K.epsilon()) #cross_entropy
        # fl = alpha * K.pow(1 - y_pred, gamma) * ce
        fl = K.pow(1 - y_pred, gamma) * ce  #多分类的alpha系数没有意义
        reduce_fl = K.sum(fl, axis=-1)
        # reduce_fl = K.max(fl, axis=-1)

        return reduce_fl
        # return (1-e) * reduce_fl + e * K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)


    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2, e=0.1):
        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)
        one_minus_p = tf.where(tf.greater(y_true, zeros), y_true - y_pred, zeros)
        ft = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(y_pred, K.epsilon(), 1.0)) #将张量y_pred真的元素压缩在[1e-8, 1]之间

        # 2# get balanced weight alpha
        # classes_num = [self.nb_neg, self.nb_pos, self.nb_gen]  # 每个类别的样例数量
        # classes_weight = tf.zeros_like(y_pred, dtype=y_pred.dtype)
        # total_num = float(sum(classes_num))
        # classes_w_t1 = [total_num / ff for ff in classes_num]
        # sum_ = sum(classes_w_t1)
        # classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        # classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=y_pred.dtype)
        # classes_weight += classes_w_tensor
        # alpha = tf.where(tf.greater(y_true, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        # balanced_fl = alpha * ft

        balanced_fl = tf.reduce_sum(ft)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        # nb_classes = len(classes_num)
        # 构造了一个均匀分布，防止过拟合
        final_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)
        return final_loss

    # 加载LSTM模型
    def load_lstm_model(self, model_path, weight_path):
        logging.info('loading models......')
        with open(model_path, 'r') as f:
            yaml_string = yaml.load(f)
        lstm_model = model_from_yaml(yaml_string)
        logging.info('loading weights......')
        lstm_model.load_weights(weight_path)
        lstm_model.compile(loss=losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"])
        # lstm_model.compile(loss=self.focal_loss_multi_class, optimizer="adam", metrics=["accuracy"])
        return lstm_model

    # 根据词向量模型得到词索引{词: 索引} 和 词向量{词: 词向量}
    def get_word_indexs(self, word2vec_model):
        if word2vec_model is not None:
            gensim_dict = Dictionary()  # {索引: 词}
            # 实现词袋模型
            gensim_dict.doc2bow(word2vec_model.wv.vocab.keys(), allow_update=True)  # (token_id, token_count)
            word2index = {wd: idx + 1 for idx, wd in gensim_dict.items()}  # 词索引字典 {词: 索引}，索引从1开始计数
            return word2index
        else:
            return None

    # 将分词序列转换成索引序列（分词序列由list列表传入）
    def text2index_from_lst(self, comm_seqs):
        data = []
        for seqs in comm_seqs:
            wd_idxs = []
            for wd in seqs:
                if wd in self.word2index.keys():
                    wd_idxs.append(self.word2index[wd])  # 单词转索引数字
                else:
                    wd_idxs.append(0)  # 索引字典里没有的词转为数字0
            data.append(wd_idxs)

        return sequence.pad_sequences(data, self.max_len)  # 对齐序列

    # 情感预测（传入评论列表）
    def predict_by_lst(self, new_comms):
        new_comms = [TextUtils.process(com) for com in new_comms if not TextUtils.is_blank(com)]
        wd_seqs = [TextUtils.tokenize(com) for com in new_comms if not TextUtils.is_blank(com)]
        X = self.text2index_from_lst(wd_seqs)
        res = []
        for i, x in enumerate(X):
            x = np.array(x).reshape(1, -1)
            lbl = self.lstm_model.predict_classes(x)  # 预测类别
            if lbl[0] == 1:
                res.append('pos')
            elif lbl[0] == 2:
                res.append('neg')
            else:
                res.append('gen')
        return res

    # 情感预测（传入文件路径）
    def predict_by_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as fin:
            # new_comms = fin.read().splitlines()
            new_coms = [line.strip() for line in fin if not TextUtils.is_blank(line)]
            res = self.predict_by_lst(new_coms)
        return res
