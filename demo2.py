# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义变量
x_data = tf.placeholder("float", [None, 784])
y_data = tf.placeholder("float", [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_hat = tf.nn.softmax(tf.matmul(x_data, w)+b)
# 定义损失函数和优化器
loss = -tf.reduce_sum(y_data * tf.log(y_hat))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# 定义准确率
correct_prediction = tf.equal(tf.argmax(y_data, 1), tf.argmax(y_hat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 定义变量初始化器
init = tf.global_variables_initializer()
# 执行训练
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        trainData = mnist.train.next_batch(1000)
        sess.run(train, feed_dict={x_data: trainData[0], y_data: trainData[1]})
        print(sess.run(accuracy, feed_dict={x_data: trainData[0], y_data: trainData[1]}),
              sess.run(loss, feed_dict={x_data: trainData[0], y_data: trainData[1]}))
    testData = mnist.test.next_batch(5000)
    print('============')
    print(sess.run(accuracy, feed_dict={x_data: testData[0], y_data: testData[1]}))


