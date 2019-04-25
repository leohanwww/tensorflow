# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:45:12 2019
定义训练
@author: leohanwww
"""

#import os
import tensorflow as tf
import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEP = 30000
MODEL_SAVE_PATH = 'd:/tensorflow/model/model.ckpt'
#MODEL_NAME = 'model.ckpt'
mnist = input_data.read_data_sets('d:/tensorflow', one_hot=True)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.infenence(x, regularizer, tf.AUTO_REUSE)
    global_step = tf.Variable(0, trainable=False)
    
    #滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #学习率动态下降法
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #用这个上下文管理器同时反向更新参数和参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)#获取batch_size个输入数据和验证数据
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
            
            if i % 1000 == 0:
                print('after %d training steps,loss on training batch is %g' % (step, loss_value))
                saver.save(sess, MODEL_SAVE_PATH, global_step=global_step)
                
def main(arvg=None):
     train(mnist)
        
if __name__ == '__main__':
    tf.app.run()