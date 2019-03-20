深层神经网络
线性模型只能做简单的区分问题，复杂问题无法通过直线划分

使用激活函数使得输出不再是线性的
tf.nn.relu
tf.sigmoid
tf.tanh

a = tf.nn.relu(tf.matmul(x,w1)+b)
y = tf.nn.relu(tf.matmul(a,w2)+b)

增加层以解决异或问题
单层神经网络无法解决异或问题（相同符号为0，不同符号为1）
深层神经网络实际上有组合特征提取的功能。这个特性对于解决不易提取特征向量的问题（比如图片识别、语音识别等）有很大帮助，通过添加一层神经网络，这层网络识别出了分界线

损失函数
分类问题的输出采用softmax层，用原神经网络的输出作为softmax的输入，把最终输出变成概率分布，从而可以使用交叉嫡来计算预测概率分布和答案之间的距离（交叉嫡就是计算两个概率分布之间距离的函数）

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10, 1.0)))
y_表示正确结果
y表示预测结果
clip_by_value是一个标准化函数，规定了下限和上限
v = tf.constant([[1.0, 2.0 , 3 .0], [4.0 , 5.0,6.0]))
print tf.clip_by_value(v, 2 .5, 4 . 5) . eval()
#输出［［ 2 .5 2 . 5 3.] [ 4. 4 . 5 4.5]]#小于2.5的取2.5，大于4.5的取4.5
tf.reduce_mean求矩阵的平均数
可以直接使用tf里的函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(lmabels=y_, logits=y)

回归问题用于预测一个数值
mse = tf.reduce_mean(tf.square(y_ - y))#均方误差损失函数

自订损失函数

import tensorf low as tf
from numpy . random import RandomState
batch_size = 8

x = tf.placeholder(tf.float32, shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,2), name='y-input')

w1 = tf.Variable(tf.random_normal([2,1],stddv=1,seed=1))
y = tf.matmul(x,w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y - y_)*loss_more,(y_ - y)*loss_less))
train_step= tf.train.AdamOptimizer(0.001).minimize(loss)
























































































