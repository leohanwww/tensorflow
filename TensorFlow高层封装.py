TensorFlow-Slim

import tensorflow as tf
import tensoflow.contrib.slim as slim
import numpy as np
from tensorflow.examples.tutorials.mnist import inpu_data

def lenet5(inputs):
	inputs = tf.reshape(inputs, [-1, 28, 28, 1])
	inputs = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')
	net = slim.max_pool2d(inputs, 2, stride=2, scope='layer2-max-pool')
	net = slim.flatten(net, scope='flaten')#从四维转换为二维
	net = slim.fully_connected(net, 500, scope='layer5')
	net = slim.fully_connected(net, 10, scope='output')
	return net

tensorflow的高层封装TFLearn

import tflearn
import tflearn.dataset.mnist as mnist
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv2d, max_pool_2d
from tflearn.layers.estimator import regression

trainX, trainY, testX, testY = mnist.load_data(path, one_hot=True)

trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

net = input_data(shape=[None, 28, 28, 1], name='input')
net = conv2d(net, 32, 5, activation='relu')#最后带激活函数
net = max_pool_2d(net, 2)
net = conv2d(net, 64, 5, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 500, activation='relu')
net = fully_connected(net, 10, activation='softmax')

net = regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')#封装过的函数定义学习任务

#网络结构训练模型,这是封装好的
model = tflearn.DNN(net, tensorboard_verbose)
#还能进行验证
model.fit(trainX, trainY, n_epoch=20, validation=([testX, testY]),show_metric=True)


keras
#使用keras实现lenet5模型
from keras.dataset import mnist
from keras.models import Sequential
from kears.layers import DENSE, Flatten, Conv2D, MxaPooling2D
from keras import backend as K

num_classes = 10
img_rows, img_cols = 28, 28
(trainX. trainY), (testX, testY) = mnist.load_data()

if K.image_data_format() == 'channels_first':
	trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
	testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
	testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows,img_cols, 1)
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0
#将标准答案转化为one-hot编码
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = kears.utils.to_categorical(testY, num_classes)

#这是keras定义的模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(MxaPooling2D(poolsize=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MxaPooling2D(poolsize=(2, 2)))
model.add(Flatten())#可以认为是个把4D-tensor拉成1D-tensor的层
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizer.SGD(), metrics=['accuracy'])#定义损失函数 优化函数 测评方法

model.fit(trainX, trainY,
		  batch_size=128,
		  epoch=20,
		  validation_data=(testX, testY))
		  
score = model.evaluate(testX, testY)

print('test loss:' score[0])
print('test accuracy:' score[1])

keras函数式api实现Inception模型

from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model

input_img = Input(shape=(28,28,1))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate(tower_1, tower_2, tower_3, axis=1)

model = Model(inputs=input_img, outputs=output)


多个输入-多个输出模型

import keras
from keras.model import Model
from keras.layers import Input, Dense
from keras.datasets import mnist

input_1 = Input(shape=(784,), name='input1')

x = Dense(1, activation='relu')(input_1)
output_1 = Dense(10, activation='softmax', name='output2')(x)

input_2 = Input(shape=(10,) name='input2')

connected = kears.layer.concatenate(x, input_2)
output_2 = Dense(10, activation='softmax', name='output2')(connected)

model = Model(inputs=[input1, input2], output=[output_1, output_2])

loss = {'output_1':binary_crossentropy, 'output_2':categorial_crossentropy}
model.compile(loss=loss,
				optimizer=keras.optimizers.SGD(),
				loss_weights = [1, 0.1]
				metrics=['accuracy'])
model.fit(
		{'input_1': trainX, 'input_2': trainY},
		{'output_1': trainY, 'output2': trainY},
		batch_size=32
		epoch=10,
		validation_data=([testX, testY], [testY, testY]))






































































































































