# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:11:40 2019
定义前向传播的参数和计算过程
@author: leohanwww
"""

import tensorflow as tf

INPUT_NODE = 784
LAYER1_NODE = 100
OUTPUT_NODE = 10

#获得变量的函数
def get_weight(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer !=None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#前向传播函数
def infenence(input_tensor, regularizer, reuse):
    #输入层-中间层
    with tf.variable_scope('layer1', reuse=reuse):
        weights = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    #中间层-输出层
    with tf.variable_scope('layer2', reuse=reuse):
        weights = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases =tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    
    return layer2