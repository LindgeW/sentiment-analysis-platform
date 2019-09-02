import socket
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource, fields, marshal_with

resource_fields = {
    'data': fields.List(fields.String, default=[]),
}


class RespEntity(object):
    def __init__(self, data):
        self.data = data



class Test(Resource):

    def __init__(self):
        self._req_param = "remarks"
        self.req_parse = reqparse.RequestParser()
        self.req_parse.add_argument(self._req_param, type=str, required=True, action='append',
                                      help='invalid params!')

    @marshal_with(resource_fields)
    def post(self):
        print(request.headers)
        print(request.method)
        print(request.url)
        print(request.json)

        args = self.req_parse.parse_args()
        print(args)
        print(args["remarks"])
        return RespEntity(data=args['remarks'])


def get_ip():
    hostname = socket.getfqdn(socket.gethostname())  # 主机名
    # hostname = socket.gethostname()
    return socket.gethostbyname(hostname)  # 内网IP


app = Flask(__name__)
api = Api(app)
api.add_resource(Test, '/test')

if __name__ == '__main__':
    ip = get_ip()
    print(ip)
    app.run(host=ip, port=500)

