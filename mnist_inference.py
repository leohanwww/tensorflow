# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#定义获取变量函数
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer = tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
#当给出正则化生成函数，将当前变量加入自定义集合losses不在tensorflow自动管理的集合列表中
    return weights

def get_biase_variable(shape):
    biases = tf.get_variable(
        "biases",shape,
        initializer = tf.constant_initializer(0.0))
    return biases

#定义向前传播过程
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE,LAYER1_NODE], regularizer)
        biases = get_biase_variable([LAYER1_NODE])
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE,OUTPUT_NODE], regularizer)
        biases = get_biase_variable([OUTPUT_NODE])
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2