import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        """get请求"""
        a = self.get_argument('a')
        b = self.get_argument('b')
        # print(a)
        # print(type(a))
        # print(b)
        # print(type(b))
        c = int(a) + int(b)

        self.write("c=" + str(c))

    def post(self):
        """get请求"""
        a = self.get_argument('a')
        b = self.get_argument('b')
        # print(a)
        # print(type(a))
        # print(b)
        # print(type(b))
        c = int(a) + int(b)

        self.write("c=" + str(c))


application = tornado.web.Application([(r"/ocr", MainHandler), ])

if __name__ == "__main__":
    application.listen(8188)
    tornado.ioloop.IOLoop.instance().start()