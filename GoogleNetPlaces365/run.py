# Import the converted model's class
from model import GoogleNetPlaces365
from PIL import Image
import tensorflow as tf
import numpy as np
import glob


def main(unused_argv):
    image = Image.open("predict/cascada.jpeg")
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    predict_data = np.array(image)
    predict_input_fn = tf.convert_to_tensor(predict_data, np.float32)
    print(predict_data.shape)
    print(predict_input_fn)
    net = GoogleNetPlaces365({'data':predict_input_fn})

    with tf.Session() as sesh:
        # Load the data
        net.load('weights.npy', sesh)
        # Forward pass
        output = sesh.run(net.get_output())
        print(output)




if __name__ == "__main__":
    tf.app.run()