from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cnn_mnist_model as model
from PIL import Image, ImageEnhance
from PIL import ImageOps
from resizeimage import resizeimage
import glob

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  # Load training and eval data
  # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  # Descomentar estas dos lineas y comentar la siguiente si se quiere introducir varias imagenes a la vez
  # filelist = glob.glob("MNIST-data/PREDICT/*.png")
  # predict_data = np.array([np.array(Image.open(fname)) for fname in filelist], dtype='f')
  
  image = Image.open("MNIST-data/PREDICT/my3.png")

  if image.size > (28, 28):
      image_28 = resizeimage.resize_thumbnail(image, [28, 28])
   #   print(np.array(image_28, dtype='f').shape)
      image_28bw = image_28.convert('L')
      brightness = ImageEnhance.Brightness(image_28bw)
      image_28bw_b = brightness.enhance(1.1)
      image_28bw_b_invert = ImageOps.invert(image_28bw_b)
      contrast = ImageEnhance.Contrast(image_28bw_b_invert)
      image_TOTAL = contrast.enhance(10)
      print(np.array(image_TOTAL, dtype='f'))

      aa = np.uint8(image_TOTAL)
      im = Image.fromarray(aa)
      im.save('MNIST-data/PREDICT/ejecutado3.png')

      predict_data= np.array(image_TOTAL, dtype='f')

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model.cnn_model_fn, model_dir="mnist_convnet_model")

  # Comentadas estas lineas para ejecutar solo el modo PREDICT
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

  #contador = 0 ", filelist[contador], "
  for pred_dict in predict_results:
    goal_class = pred_dict['classes']
    print("La imagen es un", goal_class, "con probabilidad del", pred_dict['probabilities'][goal_class]*100, "%")
   # contador = contador + 1

if __name__ == "__main__":
    tf.app.run()
