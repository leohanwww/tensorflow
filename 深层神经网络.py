深层神经网络的特性是 多层 非线性
线性模型只能做简单的区分问题，复杂问题无法通过直线划分
如果是线性模型,单层和多层没有多大的区别y=xw1w2=x(w1w2)
使用激活函数使得输出不再是线性的
非线性激活函数
tf.nn.relu
tf.sigmoid
tf.tanh
使用激活函数去线性化
a = tf.nn.relu(tf.matmul(x,w1)+biases1)
y = tf.nn.relu(tf.matmul(a,w2)+biases2)

增加层以解决异或问题
单层神经网络无法解决异或问题（相同符号为0，不同符号为1）
深层神经网络实际上有组合特征提取的功能。这个特性对于解决不易提取特征向量的问题（比如图片识别、语音识别等）有很大帮助，通过添加一层神经网络，这层网络识别出了分界线！

损失函数
分类问题的输出采用softmax层，用原神经网络的输出作为softmax的输入，把最终输出变成概率分布，从而可以使用交叉嫡来计算预测概率分布和答案之间的距离（交叉嫡就是计算两个概率分布之间距离的函数）
H((1, 0, 0), (0.5, 0.4, 0.1)) = -(1xlog0.5 + 0 x log0.4 + 0 x log0.1)=0.3
H(1, 0, 0), (0.8, 0.1, 0.1)) = -(1xlog0.8 + 0×log0.1+0 x log0.1)=0.1
#前面是正确答案,后面是推测值

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
y_表示正确结果
y表示预测结果
clip_by_value是一个标准化函数，规定了下限和上限,在上面是标准化了y(推测值)的值,1e-10是0.0000000001,避免一些计算错误,比如log0是无效的
tf.reduce_mean求矩阵的平均数
可以直接使用tf里的函数,这个是使用softmax正态分布后计算交叉嫡的损失函数
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

回归问题用于预测一个数值,只有一个结果y,使用均方误差函数
mse = tf.reduce_mean(tf.square(y_ - y))

自订损失函数
自订损失函数可以使得神经网络拟合的结果更加接近实际问题
import tensorflow as tf
from numpy . random import RandomState
batch_size = 8

x = tf.placeholder(tf.float32, shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-iuput')
#y_是预测数量，只有一个元素的张量
w1 = tf.Variable(tf.random_normal([2,1],stddv=1,seed=1))
y = tf.matmul(x,w1)

loss_less = 10
loss_more = 1
#自定义的loss函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
#先判断y和y_的大小,true取(y - y_)loss_more,false取(y_ - y)loss_less
train_step= tf.train.AdamOptimizer(0.001).minimize(loss)

X = np.random.rand(dataset_size,2)#随机生成的输入
#用X的值的和作为正确答案Y
Y = [[xl + x2 + rdm.rand()/10.0-0.05] for (xl, x2) in X]#使用rand加入一点随机噪音
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(5000):
		start = (i * batch_size) % dataset_size 
		end = min(statr+batch_size,dataset_size)
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		print(sess.run(w1))
运行以上代码会得到W1 的值为［ 1.01934695, 1.04280889 ］，也就是说得到的预测函数
是1.02x1+ 1.04x2 ， 这要比x1+x2 大，因为在损失函数中指定预测少了的损失更大
loss less> loss more 。如果将loss less 的值调整为1, loss more 的值调整为10 ， 那么W1的值将会是［0.95525807' 0 . 9813394］ 。也就是说，在这样的设置下，模型会更加偏向于预测少一点。而如果使用均方误差作为损失函数， 那么W1 会是［ 0 . 97437561, 1.0243 33 6 ］ 。使用这个自定的损失函数会尽量让预测值离标准答案更近。通过这个样例可以感受到，对于相同的神经网络， 不同的损失函数会对训练得到的模型产生重要影响。

neural network优化算法
通过更新参数来降低loss的算法
学习率η（ learning rate）来定义每次参数更新的幅度，可以认为学习率定义的就是每次参数移动的幅度

神经网络的优化过程可以分为两个阶段，
第一个阶段先通过前向传播算法计算得到预测值，井将预测值和真实值做对比得出两者之间的差距。
第二个阶段通过反向传播算法计算损失函数对每一个参数的梯度，再根据梯度和学习率使用梯度下降算法更新每一个参数

batch法综合梯度下降法和随机梯度下降法sgd，这个算法不计算全部训练数据的损失，而是随机优化某一条训练数据上的损失函数（每次计算batch数据的损失函数）
batch_size = 8
#每次读取一小部分数据作为当前的训练数据来执行反向传播算法。
x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
	for i in range(STEPS):
		start = (i * batch_size) % dataset_size
		end =  min(start+batch_size, dataset_size)
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

学习率的问题#学习率一般都比较小，这能让参数已更小的步进更新
参数	梯度*参数*学习率（1）	更新后参数
5		2*5*1=10				5-10= -5 #参数来回震荡，无法缩小
-5		2*-5*1=10				-5+10 = 5

tf.train.exponential_decay是指数衰减法，在开始用大学习率快速减小loss，在后期减小学习率使得模型平稳
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
#decay_steps通常代表了完整的使用一遍训练数据所需要的迭代轮数。这个迭代轮数也就是总训练样本数除以每一个batch 中的训练样本数(train_data_size / batch_size)也就是epoch
global_step = tf.Variable(0)#设置global_step为一个可学习的参数
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
#在minimize函数中传入global_step将自动更新，从而学习率也将得到相应更新
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

避免过拟合
过拟合是指一个模型过于复杂，它可以很好地记忆每个训练数据中随机噪音的部分而不是要去学习训练数据中的通用趋势，如果一个模型中的参数多于所有训练数据的总和，这个模型完全可以记住所有训练数据的结果使得损失函数为0
常用方法是正则化，在损失函数中加入刻画模型复杂程度的指标，然后优化损失函数+正则化函数
w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w)
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)#带正则化的损失函数，l2_regularizer防止过度模拟训练数据中的噪音，这个函数的输出是一个数值

使用collection保存权重
#获取一层神经网络的权重，并将这个权重的L2正则化加入集合
def get_weight(shape,lambda):
	var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda)(var))#正则化var并加入集合
	return var

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
batch_size = 8
#定义每层神经元个数
layer_dimension = [2,10,10,10,1] 
cur_layer = x
in_dimension = layer_dimension[0]

for i in(1,len(layer_dimension)):
	out_dimension = layer_dimension[i]#下一个输出层的节点个数
	weight = get_weight((in_dimension,out_dimension),0.001)
    #获得权重
	b = tf.constant(0.1,shape=[out_dimension])
	cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
	in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses',mse_loss)#将本次loss加入losses集合里
loss = tf.add_n(tf.get_collection('losses'))
#正则化的本质就是loss+regular(var)

滑动平均模型
使得模型在测试数据上更加健壮robust
v1 = tf.Variable(0, dtype=tf.float32)#设置这个值以后让滑动平均来更新
step = tf.Variable(0, trainable=False)#模拟一个模型的迭代轮数
#滑动平均的类，参数是衰减率和动态控制衰减率的变量
ema = tf.train.ExponentialMovingAverage(0.99, step)
#定义一个列表，里面保存要应用滑动平均的参数
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

print(sess.run([v1,ema.average(v1)]))
>>>[0.0, 0.0]#初始都为0

sess.run(tf.assign(v1, 5))#更新参数v1的值为5
sess.run(maintain_averages_op)
sess.run([v1, ema.average(v1)])
>>>[5.0, 4.5]#v1和它的滑动平均值

sess.run(tf.assign(step,10000))
sess.run(tf.assign(v1,10))
sess.run(maintain_averages_op)
#更新vl 的滑动平均值。衰减率为min{0.99, (l+step)/(lO+step)}=0.99 ,
#所以vl 的滑动平均会被更新为0.99x4.5+0.0lxl0=4.555 。
sess.run([v1, ema.average(v1)])
>>>[10.0, 4.5549998]














