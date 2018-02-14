#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cnn_mnist_model as model
from PIL import Image
import glob

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  filelist = glob.glob("MNIST-data/PREDICT/*.png")
  predict_data = np.array([np.array(Image.open(fname)) for fname in filelist], dtype='f')

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model.cnn_model_fn, model_dir="Escritorio/checkpoints/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  # train_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": train_data},
  #     y=train_labels,
  #     batch_size=100,
  #     num_epochs=None,
  #     shuffle=True)
  # mnist_classifier.train(
  #     input_fn=train_input_fn,
  #     steps=20000,
  #     hooks=[logging_hook])

  # Evaluate the model and print results
  # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": eval_data},
  #     y=eval_labels,
  #     num_epochs=1,
  #     shuffle=False)
  # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  # print(eval_results)

  #Making a predict
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": predict_data},
    shuffle=False)
  predict_results = mnist_classifier.predict(input_fn=predict_input_fn)

  contador = 0
  for pred_dict in predict_results:
    goal_class = pred_dict['classes']
    print("La imagen", filelist[contador], "es un", goal_class, "con probabilidad del", pred_dict['probabilities'][goal_class]*100, "%")
    contador = contador + 1

if __name__ == "__main__":
    tf.app.run()