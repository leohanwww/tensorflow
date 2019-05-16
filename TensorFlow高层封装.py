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


Estimator

#需要指定神经网络的输入层,所有这里指定的输入会拼接在一起作为整个神经网络的输入
feature_columns = [tf.feature_column.numeric_column("image", shape=[784])]

estimator = tf.estimator.DNNClassfier(#多个全连接层结构
	feature_columns=feature_columns,
	hidden_units=[500],#结构
	n_classes=10,#类目
	optimizer=tf.train.AdamOptimizer(),
	model_dir="/path"#将训练过程中的loss变化及其他指标保存到此

#定义训练时的数据输入
train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"image": mnist.train.images},
	y=mnist.train.labels.astype(np.int32),
	num_epochs=None,
	batch_size=128,
	shuffle=True)
	
#训练模型
estimator.train(input_fn=train_input_fn, steps=10000)

#测试时数据输入
test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"image": mnist.test.images},
	y=mnist.test.labels.astype(np.int32),
	num_epochs=1,
	batch_size=128,
	shuffle=False)
	
accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
print(accuracy_score)


自定义estimator模型

def lenet(x, is_trainning):#前向传播函数
	......
	return net
	
def model_fn(features, labels, mode, params):
	#得到前向传播结果
	predict = lenet(
		features["image"], mode == tf.estimator.ModeKeys.TRAIN)
	#如果在预测模式,只需返回结果
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions={"result": tf.argmax(predict, 1)})
	#自定义损失函数
	loss = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=predict, labels=labels))
	#自定义优化函数
	optimizer = tf.train.GradientDesentOptimaizer(
		learning_rate=params["learning_rate"])
	#定义训练过程
	train_op = optimizer.minimize(
		loss=loss, global_step=tf.train.get_global_step())
	#定义验证
	eval_metric_ops = {
		"my_metric": tf.metrics.accuracy(
			tf.argmax(predict, 1), labels)}
	#返回模型训练使用的参数
	return tf.estimator.EstimatorSpec(
		mode=mode,
		loss=loss,
		train_op=train_op,
		eval_metric=eval_metric_ops)

#通过自定义方式生成Estimator类		
model_params = {"learning_rate": 0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

train_input_fn = ...
estimator.train(input_fn=train_input_fn, steps=10000)
test_input_fn = ...
test_input_fn = ...
test_results = estimator.evaluate(input_fn=test_input_fn)
accuracy_score = test_results['metric']

#使用训练过的数据在新数据上预测
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={'image': mnist.test.images},
	num_epochs=1,
	shuffle=False)
predictions = estimator.predict(input_fn=predict_input_fn)
fro i, p in enmuerate(predictions):
	print('prediction %d: %s' % (i, p['result']))










































































































