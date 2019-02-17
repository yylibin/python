# coding=utf-8

import numpy as np
import tensorflow as tf

# x_data = np.random.rand(100).astype(dtype=np.float32)
# y_data = x_data*5.5+3.4
x_data = tf.placeholder(tf.float32)
y_data = tf.placeholder(tf.float32)
# 定义变量
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(0.5)
y_hat = w*x_data+b
# 定义损失函数和优化器
loss = tf.reduce_mean(tf.square(y_hat-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(loss)
# 定义变量初始化器
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(20000):
        x = np.random.rand(100).astype(dtype=np.float32)
        sess.run(train, feed_dict={x_data: x, y_data: x*5.5+3.4})
        if step % 100 == 0:
            print(step, sess.run(w), sess.run(b))





