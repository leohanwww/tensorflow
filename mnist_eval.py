# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:17:50 2019

@author: leohanwww
"""
import time
import tensorflow as tf
import mnist_inference
import mnist_train
from tensorflow.examples.tutorials.mnist import input_data

def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    y = mnist_inference.infenence(x, None, True)
    #计算forword的正确率
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    #保存了参数的影子变量,其实就是只保存了滑动平均值,舍弃了真实值
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    
    while True:
        with tf.Session() as sess:
            #找到checkpoint文件加载最新的模型
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt:
                #加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                #获得模型里保存的迭代轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x:mnist_train.mnist.validation.images, \
                                                               y:mnist_train.mnist.validation.labels})
                print('after %s trainnig steps ,validation' 'accuracy = %g' %(global_step, accuracy_score))
            else:
                print('no checkpoint data found')
                return
            time.sleep(10)
            
def main(arvg=None):
    evaluate(mnist_train.mnist)
    
if __name__ == '__main__':
    tf.app.run()