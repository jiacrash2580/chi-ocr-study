# coding=utf-8
import os
import shutil
import sys
import time
import tornado.ioloop
import tornado.web

import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

import crnn_test

sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])
    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16
    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        """get请求"""
        if os.path.exists(FLAGS.output_path):
            shutil.rmtree(FLAGS.output_path)
        os.makedirs(FLAGS.output_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)

                crnn_model = crnn_test.init_crnn()

                im_fn = get_images()[0]

                print('===============')
                print(im_fn)

                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except Exception as e:
                    print("Error reading image {}!".format(im_fn))
                    print(str(e))
                    return "error"

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob], feed_dict={input_image: [img], input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')

                ctpn_start = time.time()

                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                print("ctpn cost time: {:.2f}s".format(time.time() - ctpn_start))
                crnn_start = time.time()

                # select matrix [left,top,right,bottom]
                boxes = boxes[:, [0, 1, 4, 5]]
                # sort boxes by top value
                boxes = boxes[boxes[:, 1].argsort()]

                ocr_result = "";
                last_line_v = [];
                for i, box in enumerate(boxes):
                    if i > 0:
                        if (last_line_v[1] - box[1]) * 3 < box[3] - last_line_v[0]:
                            ocr_result += '\r\n'
                    else:
                        last_line_v = [box[1], box[3]]
                    imgline = img[box[1]:box[3], box[0]:box[2]]
                    pil_image = Image.fromarray(imgline)
                    ocr_result += crnn_test.crnn_recognition(pil_image, crnn_model)

                print('results: {0}'.format(ocr_result))
                print("crnn cost time: {:.2f}s".format(time.time() - crnn_start))
                self.write(ocr_result)

    def post(self):
        """post请求"""
        self.write("post")


application = tornado.web.Application([(r"/ocr", MainHandler), ])


if __name__ == '__main__':
    application.listen(8188)
    tornado.ioloop.IOLoop.instance().start()
