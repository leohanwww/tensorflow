# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:44:46 2019

@author: zhanghan
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('d:/tensorflow',one_hot=True)

x = tf.placeholder(tf.float32,[None,784],name='x-input')
y_ = tf.placeholder(tf.float32,[None,10],name='y-input')

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

cross_enpotry = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_enpotry)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    
    for i in range(10000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(accurary,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))