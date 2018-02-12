#Importamos los datos
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#Creamos los placeholder y las variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#En y_ almacenaremos los datos correctos
y_ = tf.placeholder(tf.float32, [None, 10])

#Funcion que almanacena el error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#Entrenamiento (Minimizar el error)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Define la operacion para incializar las variables creadas
init = tf.global_variables_initializer()

#Inicializamos las variables y arrancamos
sess = tf.Session()
sess.run(init)

#Lo entrenaremos 1000 veces, y para hacerlo mas ligero, cada vez solo cogeremos 100 datos
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#Evaluamos el acierto del programa (True o False)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Los valores True o False los convertimos a 1 o 0 respectivamente
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Imprimimos la precisión
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


