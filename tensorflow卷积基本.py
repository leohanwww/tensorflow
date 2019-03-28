tensorflow基础

Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它
如果没有interactivesession，需要在启动session之前构建整个计算图，然后启动它，
使用interactivesession，可以在运行图的时候，动态插入一些计算图
import tensorflow as tf
sess = tf.InteractiveSession()
sess.run(...)
我们用python描绘计算图，然后在python之外运行，并且可以定义哪一部分运行，此时需要session


构建多层卷积网络

def weight_variable(shape):
	initial = tf.truncate_normal(shape, stddev=0.1)
	return tf.Variable(initial)
#这两个函数根据具体shape构建参数
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

#卷积
def con2d(x, W):
	return tf.nn.con2d(x, W, strides=[1,1,1,1],padding='SAME')
#池化
def max_pool_2x2(x):
	return tf.nn.pool(x, ksize=[1,2,2,1],strides=[1,2,2,3],padding='SAME')

#开始实施
#卷积在每个5x5的patch中算出32个特征
#前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])#4d向量，宽28、高28、通道数为1代表灰度图

h_conv1 = tf.nn.relu(con2d(x_image,W_conv1) + b_con1)#应用卷积层
h_pool1 = max_pool_2x2(h_con1)#应用池化层

#第二层卷积
W_conv2 = weight_variable([5,5,32,64])#第二层得到64个特征
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(con2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])#缩小图片尺寸到7*7，加入1024个神经元的全连接层
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#Dropout层
#减少过拟合,这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#测试模型
croee_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
	batch_x, batch_y = mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accurary = accuracy.eval(feed_dict={x:batch_x, y_:batch_y,keep_prob: 1.0})
		print('after %d training steps, the accuracy is %g', % (i, train_accurary))
	train_step.run(feed_dict={x:batch_x, y_:batch_y, keep_prob: 0.5})
print('test accuracy %g' %accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

#卷积函数
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

参数：
**input : ** 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

import tensorflow as tf
# case 1
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 1*1 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))
op1 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

# case 8  
# 输入是10 张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后每张图得到7个 3*3 的feature map (考虑边界,不足的地方用0填充)
# 10张图最后输出就是一个 shape为[10,3,3,7] 的张量
input = tf.Variable(tf.random_normal([10,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')  

计算卷积后大小的公式

1.卷积后尺寸计算 
out_height=(in_height+2pad-filter_height)/strides[1]+1 
out_width=(in_width+2pad-filter_width)/strides[2] +1 
2.tensorflow中卷积参数same和valid运算之后的维度计算 
(1)same 
out_height=ceil(float(in_height))/float(strides[1]) #ceil是小数向上取整
out_width=ceil(float(in_width))/float(strides[2]) 
(2)valid 
out_height=ceil(float(in_height-filter_height+1))/float(strides[1]) 
out_width=ceil(float(in_width-filter_width+1))/float(strides[2]) 


























