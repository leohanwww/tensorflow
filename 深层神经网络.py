深层神经网络
线性模型只能做简单的区分问题，复杂问题无法通过直线划分

使用激活函数使得输出不再是线性的
非线性激活函数
tf.nn.relu
tf.sigmoid
tf.tanh
使用激活函数去线性化
a = tf.nn.relu(tf.matmul(x,w1)+b)
y = tf.nn.relu(tf.matmul(a,w2)+b)

增加层以解决异或问题
单层神经网络无法解决异或问题（相同符号为0，不同符号为1）
深层神经网络实际上有组合特征提取的功能。这个特性对于解决不易提取特征向量的问题（比如图片识别、语音识别等）有很大帮助，通过添加一层神经网络，这层网络识别出了分界线！

损失函数
分类问题的输出采用softmax层，用原神经网络的输出作为softmax的输入，把最终输出变成概率分布，从而可以使用交叉嫡来计算预测概率分布和答案之间的距离（交叉嫡就是计算两个概率分布之间距离的函数）

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10, 1.0)))
y_表示正确结果
y表示预测结果
clip_by_value是一个标准化函数，规定了下限和上限
v = tf.constant([[1.0, 2.0 , 3 .0], [4.0 , 5.0,6.0]))
print tf.clip_by_value(v, 2.5, 4.5) . eval()
#输出［［ 2 .5 2 . 5 3.] [ 4. 4 . 5 4.5]]#小于2.5的取2.5，大于4.5的取4.5
tf.reduce_mean求矩阵的平均数
可以直接使用tf里的函数,这个是使用softmax正态分布后计算交叉嫡的损失函数
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

回归问题用于预测一个数值,只有一个结果y
mse = tf.reduce_mean(tf.square(y_ - y))#均方误差损失函数

自订损失函数
自订损失函数可以使得神经网络拟合的结果更加接近实际问题
import tensorf low as tf
from numpy . random import RandomState
batch_size = 8

x = tf.placeholder(tf.float32, shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-iuput')
#y_是预测数量，只有一个元素的张量
w1 = tf.Variable(tf.random_normal([2,1],stddv=1,seed=1))
y = tf.matmul(x,w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y - y_)*loss_more,(y_ - y)*loss_less))#这个loss取(算多了的和算少了的和算的正好)的平均值
train_step= tf.train.AdamOptimizer(0.001).minimize(loss)

X = np.random.rand(dataset_size,2)
Y = [[xl + x2 + rdm.rand()/10.0-0.05] for (xl, x2) in X]#使用rand加入一点随机噪音
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(5000):
		start = (i * batch_size) % dataset_size 
		end = min(statr+batch_size,dataset_size)
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		print(sess.run(w1))
运行以上代码会得到W1 的值为［ 1.01934695, 1.04280889 ］，也就是说得到的预测函数
是I .02 x 1+ l . 04x2 ， 这要比x1+x2 大，因为在损失函数中指定预测少了的损失更大
loss less> loss more 。如果将loss less 的值调整为1, loss more 的值调整为10 ， 那么W1的值将会是［0.95525807' 0 . 9813394］ 。也就是说，在这样的设置下，模型会更加偏向于预测少一点。而如果使用均方误差作为损失函数， 那么W1 会是［ 0 . 97437561, 1.0243 33 6 ］ 。使用这个损失函数会尽量让预测值离标准答案更近。通过这个样例可以感受到，对于相同的
神经网络， 不同的损失函数会对训练得到的模型产生重要影响。

通过更新参数来降低loss的算法
学习率η（ learning rate）来定义每次参数更新的幅度，可以认为学习率定义的就是每次参数移动的幅度

神经网络的优化过程可以分为两个阶段，第一个阶段先通过前向传播
算法计算得到预测值，井将预测值和真实值做对比得出两者之间的差距。然后在第二个阶
段通过反向传播算法计算损失函数对每一个参数的梯度，再根据梯度和学习率使用梯度下
降算法更新每一个参数

batch法综合梯度下降法和随机梯度下降法，每次计算batch数据的损失函数
batch_size = n
#每次读取一小部分数据作为当前的训练数据来执行反向传播算法。
x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y = tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')
loss = 
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	for i in range(STEPS):
		start =...
		end =...  
		sess.run(train_step,feed_dict={x:X[start:end],y:Y[start:end]})

学习率的问题
参数		梯度*学习率（1）	更新后参数
5		2*5*1=10		5-10= -5 #参数来回震荡，无法缩小
-5		2*-5*1=10		-5+10 = 5

tf.train.exponential_decay是指数衰减法，在开始用大学习率快速减小loss，在后期减小学习率使得模型平稳
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
#指定初始学习率为0.1，每训练100轮后学习率乘以0.96
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

避免过拟合
w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w)
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)#带正则化的损失函数

#获取一层神经网络的权重，并将这个权重的L2正则化加入集合
def get_weight(shape,lambda):
	var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda)(var))
	return var

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
batch_size = 8
#定义每层神经元个数
layer_dimension = [2,10,10,10,1]
layers_lenth = len(layer_dimension)
cur_layer = x
in_dimension = layer_dimension[0]

for i in(1,layers_lenth):
	out_dimension = layer_dimension[i]#构建下一个函数的参数
	weight = get_weight((in_dimension,out_dimension),0.001)
#获得权重
	b = tf.constant(0.1,shape=[out_dimension])
	cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + b)
	in_dimension = layer_dimension[1]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses',mse_loss)#将本次loss加入losses集合里
loss = tf.add_n(tf.get_collection('losses'))#将所有loss相加



































