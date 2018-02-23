from __future__ import print_function
import tensorflow as tf
import argparse

from tensorcv.dataflow.image import ImageFromFile

import utils.setup_env as conf
from model.googlenet import GoogleNet
from utils.classes import get_word_list
from utils.preprocess import resize_image_with_smallest_side




def display_data(dataflow, data_name):
    try:
        print('[{}] num of samples {}, num of classes {}'.
              format(data_name, dataflow.size(), len(dataflow.label_dict)))
    except AttributeError:
        print('[{}] num of samples {}'.
              format(data_name, dataflow.size()))
    print(dataflow._im_list)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default='.jpg', type=str,
                        help='image file extension')
    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = get_args()

    model = GoogleNet(is_load=True, pre_train_path='googlenet.npy')

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    test_data = ImageFromFile(FLAGS.type,
                              data_dir=conf.DATA_DIR,
                              num_channel=3)
    word_dict = get_word_list('data/imageNetLabel.txt')

    model.create_model([image, 1])
    test_op = tf.nn.top_k(tf.nn.softmax(model.layer['output']),
                          k=5, sorted=True)
    input_op = model.layer['input']

    writer = tf.summary.FileWriter(conf.SAVE_DIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        a = 0
        for k in range(0, 50):
            if test_data.epochs_completed < 1:
                print("***** Imagen:", a + 1, '*****  --> ', test_data._im_list[a], a)
                batch_data = test_data.next_batch()
                im = batch_data[0]
                im = resize_image_with_smallest_side(im, 224)
                result = sess.run(test_op, feed_dict={image: im})
                for val, ind in zip(result.values, result.indices):
                    a = a + 1
                    for i in range(0,5):
                            print("Con un ",val[i],"% es un ", word_dict[ind[i]], sep='')
                    print("******************************")