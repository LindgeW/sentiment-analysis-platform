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
import warnings
import keras
# import logging
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
#from collections import Counter
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers import Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D, Convolution1D
from keras.layers.pooling import GlobalMaxPool1D, MaxPooling1D, GlobalAvgPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential, model_from_yaml
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import metrics
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import cohen_kappa_score #kappa系数
from log.logger import MyLogger
from models.text_utils import TextUtils
from keras.engine.topology import Layer
from keras import initializers
#设置日志级别，默认是logging.WARNING，低于该级别的不会输出，级别排序:CRITICAL>ERROR>WARNING>INFO>DEBUG
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# logging.basicConfig(filename='log/new.log', #日志文件
#                     filemode='a', #模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志，a是追加模式，默认是追加模式
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', #日志格式
#                     level=logging.DEBUG,#控制台打印的日志级别
#                 ) #只会保存log到文件，不会输出到控制台

#随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数
np.random.seed(3347)
#积极-1、中立-0、消极- -1
LABEL_POS, LABEL_NEG, LABEL_GEN = 1, -1, 0
N_ROWS = 20000 #每个类别的评论数
N_ITER = 8  #词向量训练的迭代次数
WIN_SIZE = 8  #词向量上下文最大距离，一般取[5-10]
MIN_COUNT = 5 #需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词
EMBEDDING_SIZE = 300 #词向量的维度，数据集越大该值越大
HIDDEN_LAYER_SIZE = 128 #隐层的大小
BATCH_SIZE = 64 #批的大小
NUM_EPOCHS = 10 #LSTM样本训练的轮数
TEST_SIZE = 0.2 #总样本中，测试样本的占比
CPU_COUNT = multiprocessing.cpu_count() #cpu线程数量

COMMENT_DIR = 'comment' #源评论文件目录
MODEL_DIR = 'model_lstm' #保存模型目录
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm.yml') #保存LSTM模型结构的路径
LSTM_WEIGHT_PATH = os.path.join(MODEL_DIR, 'lstm.h5') #保存LSTM权重的路径
# WORD2VEC_PATH = os.path.join(MODEL_DIR, 'word2vec_model.pkl') #保存word2vec词向量模型的路径
WORD2VEC_PATH = os.path.join(MODEL_DIR, 'word2vec.model') #保存word2vec词向量模型的路径
HIST_PATH = os.path.join(MODEL_DIR, 'hist.pkl') #保存训练过程中的历史记录
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'model-{epoch:02d}-{val_acc:.2f}.h5') #保存训练最好的模型的权值


class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(Attention, self).__init__(**kwargs)

    # 定义权重
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            name='att_w',
            shape=(input_shape[1], input_shape[1]),
            initializer=self.init,
            trainable=True)

        self.b = self.add_weight(
            name='att_b',
            shape=(input_shape[1],),
            initializer=self.init,
            trainable=True)

        super(Attention, self).build(input_shape)

    # 编写层的功能逻辑
    def call(self, inputs, **kwargs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))  # (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    # 如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class SentimentModel:
    def __init__(self):
        self.nb_classes = 3 #3分类
        self.max_len = 100  # 文本保留的最大长度
        self.nb_pos = 0 #正面样例数
        self.nb_neg = 0 #负面例数
        self.nb_gen = 0 #中立样例数

        self.logging = MyLogger()  # 自定义日志器

        self.logging.info('###该模型为情感%d分类模型###'%(self.nb_classes))
        self.logging.info('训练日期：'+time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.logging.info('CPU线程数：{}'.format(CPU_COUNT))


    '''
    input_csv_dir：csv格式的评论数据集目录
    返回值：DataFrame对象
    '''
    def load_train_data(self, input_csv_dir):
        self.logging.info('开始加载数据......')
        t1 = time.time()
        df_data = pd.DataFrame()
        for f in os.listdir(input_csv_dir):
            csv_file = os.path.join(input_csv_dir, f)
            csv_data = pd.read_csv(csv_file, usecols=['comment', 'label'], nrows=N_ROWS)
            df_data = df_data.append(csv_data, ignore_index=True)

        df_data.dropna(inplace=True)
        df_data.drop_duplicates(subset=['comment'], inplace=True)
        df_data['segs'] = df_data['comment'].apply(TextUtils.tokenize)
        t2 = time.time()
        self.logging.info('数据加载完成！总用时：{}s'.format(t2 - t1))
        # self.max_len = max(df['segs'].apply(lambda x: len(x)))  # 序列的最大长度

        self.logging.info('数据集总记录数：{}'.format(len(df_data)))
        self.nb_pos = len(df_data[df_data['label'] == LABEL_POS]) #正样例数
        self.nb_neg = len(df_data[df_data['label'] == LABEL_NEG]) #负样例数
        self.nb_gen = len(df_data[df_data['label'] == LABEL_GEN]) #中立例数
        self.logging.info('正面例数：{} 负面样例数：{} 中立样例数：{}'.format(self.nb_pos, self.nb_neg, self.nb_gen))
        return shuffle(df_data)  #随机打乱数据集


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


    #构建LSTM模型
    def build_lstm_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(GlobalAvgPool1D())
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation="softmax"))

        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model.summary()
        return model

    # 构建LSTM模型
    def build_attention_lstm_model(self, embedding_weights):
        model = Sequential()
        vocab_size = len(embedding_weights)
        # 嵌入层将正整数（下标）转换为具有固定大小的向量. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        model.add(Embedding(input_dim=vocab_size,  # 字典长度
                            output_dim=EMBEDDING_SIZE,
                            input_length=self.max_len,
                            weights=[embedding_weights]))  # (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE), where None is the batch dimension
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(HIDDEN_LAYER_SIZE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        #     model.add(GRU(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Attention())
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation="softmax"))

        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
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
        model.add(Dense(self.nb_classes, activation="softmax"))

        # inputs = Input(shape=(None,))
        # embedded = Embedding(vocab_size, EMBEDDING_SIZE, input_length=self.max_len)(inputs)
        # lstm_out = LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(embedded)
        # predict = Dense(1, activation='softmax')(lstm_out)
        # model = Model(inputs=inputs, outputs=predict)
        model.summary()
        return model

    #多分类f1_score
    class f1_score(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []

        def on_epoch_end(self, epoch, logs={}):
            # val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
            val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
            # val_targ = self.validation_data[1]
            val_targ = np.argmax(self.validation_data[1], axis=1)

            #macro: 对每一类别的f1_score进行简单算术平均（unweighted mean）
            _val_f1 = f1_score(val_targ, val_predict, average='macro') #所有类都同样重要
            #weighted: 对每一类别的f1_score进行加权平均，权重为各类别数在y_true中所占比例
            # _val_f1 = f1_score(val_targ, val_predict, average='weighted')

            # _val_recall = recall_score(val_targ, val_predict)
            # _val_precision = precision_score(val_targ, val_predict)
            self.val_f1s.append(_val_f1)
            # self.val_recalls.append(_val_recall)
            # self.val_precisions.append(_val_precision)
            # print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
            print(' — val_f1:', _val_f1)
            return


    #构建并训练LSTM模型
    def train_lstm_model(self, embedding_weights, X, y):
        lstm_model = self.build_lstm_model(embedding_weights)
        lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

        # cvscores = [] #交叉验证结果
        # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=47)
        # for train_index, test_index in skf.split(X, y):
        #     Xtrain, Xtest = X[train_index], X[test_index]
        #     ytrain, ytest = y[train_index], y[test_index]
        #     hist = lstm_model.fit(Xtrain, ytrain,
        #                           batch_size=BATCH_SIZE,
        #                           epochs=NUM_EPOCHS,
        #                           validation_data=0.1)
        #     #第一维loss，第二维acc
        #     scores = lstm_model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
        #     print('%s: %.2f%%' % (lstm_model.metrics_names[0], scores[0] * 100))
        #     print('%s: %.2f%%'% (lstm_model.metrics_names[1], scores[1] * 100))
        #     cvscores.append(scores[1]*100)
        # cvscores = np.asarray(cvscores)
        # print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
        ytrain = keras.utils.to_categorical(ytrain, num_classes=self.nb_classes)
        ytest = keras.utils.to_categorical(ytest, num_classes=self.nb_classes)

        self.logging.info('开始训练LSTM模型......')
        self.logging.info('训练集大小：{} 测试集大小：{}'.format(round((1-TEST_SIZE)*len(X)), round(len(X)*TEST_SIZE)))
        t1 = time.time()

        #宏F1_score
        f1 = self.f1_score()
        #保存训练最好的模型（每3轮训练检测一次）
        cp = ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_acc', mode='max', save_best_only=True, verbose=1, period=2)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=NUM_EPOCHS/2)  # 经过NUM_EPOCHS/2轮训练，当val_acc不再提升，则停止训练
        callbacks_lst = [es, cp, f1]
        hist = lstm_model.fit(Xtrain, ytrain,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=callbacks_lst,
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

        # self.logging.info('metrics names:'+lstm_model.metrics_names)
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
    def load_lstm_model(self, model_path=LSTM_MODEL_PATH, weight_path=LSTM_WEIGHT_PATH):
        self.logging.info('loading models......')
        with open(model_path, 'r') as f:
            yaml_string = yaml.load(f)
        lstm_model = model_from_yaml(yaml_string)
        self.logging.info('loading weights......')
        lstm_model.load_weights(weight_path)
        lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
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
            res = lstm_model.predict_classes(x) #预测类别
            if res[0] == 1:
                print(new_comms[i], '\n', 'positive!')
            elif res[0] == 2:
                print(new_comms[i], '\n', 'negative!')
            else:
                print(new_comms[i], '\n', 'neural!')


    #情感预测（传入文件路径）
    def predict_by_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as fin:
            # new_comms = fin.read().splitlines()
            new_coms = [line.strip() for line in fin if not TextUtils.is_blank(line)]
            self.predict_by_lst(new_coms)


    # 重复训练word2vec
    # 可以将更多的训练数据传递给一个已经训练好的word2vec对象，继续更新模型的参数
    def retrain_word2vec(self, new_sentences):
        #加载word2vec模型
        word2vec_model = Word2Vec.load(WORD2VEC_PATH)
        word2vec_model.train(new_sentences)
        word2vec_model.save(WORD2VEC_PATH)

    # 在原模型的基础上重新训练模型
    def retrain_model(self):
        #加载新数据集
        # x_train, y_train, x_test, y_test = load_data()
        #加载模型
        # model = self.load_lstm_model()
        #评估模型
        # model.evaluate(x_test, y_test)
        #重新训练
        # model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
        #重新评估模型
        # model.evaluate(x_test, y_test)
        #保存模型
        # model.save(path)
        pass


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
    model.train()

    new_comms = ['音质一般，手机赠送的，9元应该不值。外观不错。',
                '还不错吧，一直支持小米！支持国货',
                '性价比非常高，显示屏幕高清，处理速度快！不带鼠标。很轻，没有想象中的薄。自带的office365需要激活',
                 '非常好看！很满意，这个价钱买到质量这么好的鞋子简直太感动了，客服推荐的码数刚好，很合脚，穿起来不磨脚，防滑，耐脏，绝对值！',
                 '什么破商家。我申请换货，不让我换，硬要求我退货，我说不退，直接把我电话挂掉，服务差评',
                 '冰箱收到了，空间挺大的，冷冻室有点小，能效是三级，冷冻，冷藏，保鲜三室可以调温，送来的时候冰箱里面有点味道！送电以后就没了！总得来说这次购物不错！',
                 '发申通物流，很不靠谱，太慢了，本来就是冲着京东物流才买的，结果发的申通，下次再不买它家的了',
                 '鞋子已是第二双，朋友看到好看帮忙买的，质量杠杠的，又轻又舒服，春天到了配啥衣服都好看，出去玩也不怕累脚了，值得推荐！！！！',
                 '用了两天才过来评论的，首先电脑颜值高，四面窄观影简直不要太舒服，背光键盘也很nice，尺寸大小也很合适，开机使用反应超快的。刚拿到手就插上电@了，语音助手指导小白也能上手。值得购买！',
                 '很差，给老爸买的靴子，本来想着尽一份孝心，结果给我的感触很大，家里在县城，那么多快递可以到，偏偏发一个到不了的快递，也不跟买家核实地址，也不打招呼，哪怕跟我们核实一下，我们可以自己选择快递啊。那么大老远，还下着大雨，直接让老人去取。这次购物体验很差，很差。卖家服务态度也很差，很差']
    model.predict_by_lst(new_comms)

    # model.predict_by_file("user/samples.txt")