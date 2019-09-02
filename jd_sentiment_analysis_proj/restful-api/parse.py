# flask的RESTful扩展库
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource, fields, marshal_with
from models.EmotionModel import SentimentModel
import socket
import json
import jieba.analyse
jieba.analyse.set_stop_words('stop_words.txt')


# 数据格式化
resource_fields = {
    'code': fields.Integer(default=200),
    'msg': fields.String(default='OK'),
    'data': fields.List(fields.String, default=[]),
    # 'result': fields.String(attribute='output'),  # 重命名属性，属性output以result名字返回
}


class RespEntity(object):
    def __init__(self, code=200, msg='OK', data=None):
        self.code = code
        self.msg = msg
        self.data = data


# 在Windows下用文本编辑器创建的文本文件，
# 如果选择以UTF-8等Unicode格式保存，会在文件头（第一个字符）加入一个BOM(Byte Order Mark)标识
class EModel:
    def __init__(self):
        self.__model = SentimentModel()
        self.__model.load()

        print('test model....')
        # 在初始化加载模型之后，让 model随便执行一次 predict函数，之后再使用就不会有问题，否则predict会出错（未知解）
        self.__model.predict_by_lst(['这双鞋子很便宜，质量很好，下次还会买的！'])
        print('test done!')

    def predict(self, com_lst):
        return self.__model.predict_by_lst(com_lst)


# Flask-RESTful提供了一个用于参数解析的RequestParser类，可以很方便的解析请求中的-d参数，并进行类型转换
class Toparse(Resource):
    def __init__(self):
        print('ready......')
        self.__req_param = 'remarks'
        self.__req_parse = reqparse.RequestParser()
        # 要求一个值传递的参数，只需要添加 required=True
        # 如果指定了help参数的值，在解析的时候当类型错误被触发的时候，它将会被作为错误信息给呈现出来
        # 接受一个键有多个值的话，你可以传入 action='append'
        self.__req_parse.add_argument(self.__req_param, type=str, required=True, action='append', help='invalid params!')

        # params_lst = ['msg', 'code', 'dataset']
        # for param in params_lst:
        #     self.__req_parse.add_argument(param, type=str, required=True, action='append', help='invalid params!')

    @marshal_with(resource_fields)
    def get(self):
        print("visiting....")
        return RespEntity(code=200, msg='Connected!')

    @marshal_with(resource_fields)
    def post(self):
        print(request.headers)
        print(request.method)
        print(request.url)
        if not request.json:
            abort(400, message='no json format!')
            return RespEntity(code=400, msg='no json format!')

        print(request.json)

        # print(request.json.get('dataset'))
        # return {'remark': 'positive'}  #Flask-RESTful会自动地处理转换成JSON数据格式，可以省去jsonify

        # strict如果提供未定义的参数, 那么就抛出异常
        args = self.__req_parse.parse_args(strict=True)  # 返回Python字典
        print(args)
        print(args[self.__req_param])
        global model
        result = model.predict(args[self.__req_param])  #标签列表
        print(result)
        return RespEntity(data=result)


class WordCloud(Resource):
    def __init__(self):
        print('ready......')
        self.__req_param = 'remarks'
        self.__req_parse = reqparse.RequestParser()
        # 要求一个值传递的参数，只需要添加 required=True
        # 如果指定了help参数的值，在解析的时候当类型错误被触发的时候，它将会被作为错误信息给呈现出来
        # 接受一个键有多个值的话，你可以传入 action='append'
        # self.__req_parse.add_argument(self.__req_param, type=str, required=True, action='append', help='invalid params!')
        self.__req_parse.add_argument(self.__req_param, type=str, required=True, help='invalid params!')

        # params_lst = ['msg', 'code', 'dataset']
        # for param in params_lst:
        #     self.__req_parse.add_argument(param, type=str, required=True, action='append', help='invalid params!')

    # TF-IDF算法提取关键字
    def get_key_words(self, contents):
        wds = []
        dicts = {}
        tags = jieba.analyse.tfidf(contents, topK=100, withWeight=True)
        # tags = jieba.analyse.textrank(contents, topK=100, withWeight=True)
        for wd, weight in tags:
            dicts['name'] = wd
            dicts['value'] = round(weight * 10000)
            wds.append(json.dumps(dicts))  # 字典转str
        return wds

    @marshal_with(resource_fields)
    def get(self):
        return RespEntity(code=404, msg='Not Found!!!')

    @marshal_with(resource_fields)
    def post(self):
        if not request.json:
            abort(400, message='no json format!')
            return RespEntity(code=400, msg='no json format!')
        # strict如果提供未定义的参数, 那么就抛出异常
        args = self.__req_parse.parse_args(strict=True)  # 返回Python字典
        remark_str = args[self.__req_param]
        keywords = self.get_key_words(remark_str)
        return RespEntity(data=keywords)


def get_ip():
    # hostname = socket.getfqdn(socket.gethostname())  # 主机名
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)  # 内网IP


model = EModel()
app = Flask(__name__)
api = Api(app)
# 设置路由urls:
# http://127.0.0.1:5000/to_parse
# http://127.0.0.1:5000/to_parse/
api.add_resource(Toparse, '/to_parse', '/to_parse/') #一个资源挂载在多个路由上
api.add_resource(WordCloud, '/wordcloud', '/wordcloud/')


if __name__ == '__main__':
    ip = get_ip()
    print(ip)
    app.run(host=ip, port=5000)
