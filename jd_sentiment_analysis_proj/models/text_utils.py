import numpy as np
import os
import re
import html
import jieba
import jieba.analyse
import pandas as pd
from snownlp import SnowNLP

USER_DIR = '../user' #用户自定义文件目录
USER_DICT_PATH = os.path.join(USER_DIR, 'user_dict.txt') #自定义用户表
STOP_WORDS_PATH = os.path.join(USER_DIR, 'stop_words.txt') #停用词表
jieba.load_userdict(USER_DICT_PATH)
# jieba.analyse.set_stop_words(STOP_WORDS_PATH)
# pos_tags = ['n', 'vn', 'v', 'ad', 'a', 'e', 'y'] #是名词、形容词、动词、副词、叹词、语气词

#自定义文本处理工具类
class TextUtils:
    __min_len = 8  #评论文本长度至少为8

    # 加载停用词表
    @classmethod
    def stop_words(cls, stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as f:
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
    def remove_blank(cls, _str):  # 去除空白字符
        return re.sub(r'\s', '', _str)

    @classmethod
    def html_unescape(cls, _str):  # 将字符串中的html实体转换成html标签
        return html.unescape(_str)

    @classmethod
    def simplified(cls, _str): #繁体中文转简体中文
        return SnowNLP(_str).han

    @classmethod
    def two_classifier(cls, _str):  #二分类
        try:
            _str = str(_str)
            if cls.is_blank(_str.strip()):
                return -100
            else:
                if len(_str) > 100:
                    idx = SnowNLP(' '.join(jieba.analyse.textrank(_str))).sentiments
                else:
                    idx = SnowNLP(_str).sentiments  #情感系数
                if idx > 0.9: #正向
                    return 1
                elif idx < 0.1: #负向
                    return -1
                else:
                    return 0
        except:
            return np.inf

    @classmethod
    def maketrans(cls, _str, src, target):  # 将源串中需要转换的字符（串）转换成目标字符（串）
        if src in _str and len(src) == len(target):
            trans_table = str.maketrans(src, target)  # 如：贼贵 -> 很贵
            return _str.translate(trans_table)
        else:
            return _str

    @classmethod
    def tokenize(cls, _str):  # 对评论列表进行分词
        return [wd for wd in jieba.lcut(_str) if wd not in cls.stop_words(STOP_WORDS_PATH)]

    @classmethod
    def normalize(cls, _str):  # 将数字和字母进行归一化，替换成统一字符 #
        pattern = re.compile(r'\w+', re.A | re.M)
        _str = re.sub(pattern, '#', _str)
        return _str

    @classmethod
    def del_repeat_elem_from_list(cls, _list):  # 删除列表中相邻重复元素，如：[1,2,2,3,3,3,4,4,5]->[1,2,3,4,5]
        result = []
        for item in _list:
            size = len(result)
            if size == 0 or result[size - 1] != item:
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
    def process(cls, sent):  # 评论句子的预处理
        sent = cls.remove_blank(sent)
        sent = cls.html_unescape(sent)
        sent = cls.simplified(sent)
        sent = cls.normalize(sent)
        sent = cls.del_repeat_words_from_str(sent)
        if cls.is_blank(sent) or len(sent) < cls.__min_len: #忽略长度过短的评论（没有多大意义）
            return ''  # 定义为空缺数据
        return sent

    # 处理csv数据集
    @classmethod
    def process_csv_data(cls, csv_file, label=0):
        df = pd.read_csv(csv_file, usecols=['comment', 'label'], encoding='utf-8')  #读取comment和star列数据
        df.dropna(inplace=True)  # 任意一行有某个属性值为NA则删除该行
        df.drop_duplicates(subset=['comment'], inplace=True)  # 删除comment列有重复的行
        df['comment'] = df['comment'].apply(TextUtils.process)
        # df['label'] = df['comment'].apply(TextUtils.two_classifier)
        df.dropna(inplace=True)
        df['segs'] = df['comment'].apply(lambda x: ' '.join(TextUtils.tokenize(x)))
        # df['label'] = label

        df.to_csv('data/gen_data.csv', columns=['label', 'segs'], index=False)

        return df


if __name__ == '__main__':
    # TextUtils.process_csv_data('neg_clear.csv')
    # TextUtils.process_csv_data('pos_clear.csv')
    TextUtils.process_csv_data('gen_clear.csv')