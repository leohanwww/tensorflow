tf入门

计算图
import tensorflow as tf
a = tf.constant([1.0, 2.0], name =”a ”)
b = tf.constant([2.0, 3.0], name =”b ”)
result = a + b
TensorFlow会自动将定义的计算转化为计算图上的节点。在TensorFlow 程序中，系统会自
动维护一个默认的计算图，通过tf.get_default_graph 函数可以获取当前默认的计算图。

g1 = tf.Graph()#通过tf.Graph函数来生成新的计算图。不同计算图上的张量和运算都不会共享
with g1.as_default():
	#在计算图g1中定义变量v并设置初始值为0
	v = tf.get_variable(
		"v", initializer=tf.zeros([1,]))

g2 = tf.Graph()#又一个计算图
with g2.as_default():
	v = tf.get_variable(
		"v", initializer=tf.ones([1,])#设置初始值为1维的1

#在计算图gl中读取变盘“v”的取值。
with tf.Session(graph=g1) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("",reuse=True):
		#输出变量v的值
		print(sess.run(tf.get_variable("v")))
>>> [0.]
with tf.Session(graph=g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))
>>> [1.]

tensorflow维护的集合
tf.GraphKeys.VARIABLES 所以变量
tf.GraphKeys.TRAINABLE_VARIABLES 可学习变量
tf.GraphKeys.SUMMARIES 日志张量
tf.GraphKeys.QUEUE_RUNNERS 
tf.GraphKeys.MOVING_AVERAGE_VARIABLES 所有计算了滑动平均值的变量



张量
Tensor("add_2:0", shape=(2,), dtype=float32)
张量三个属性:name(node:src_output),shape,type 
add_2(节点名称):0(此节点输出的第几个结果)

import tensorflow as tf
# tf.constant 是一个计算，这个计算的结果为一个张量， 保存在变量a 中。
a= tf.constant([1.0, 2.0] , name =” a ”)
b = tf.constant( [2 . 0 , 3 . 0] , name = ” b ”)
result = tf.add( a , b , name=” add ”)#创建add节点
print result
>>> Tensor( "add:0"， shape=(2,), dtype=float32)
>>> tf.Session().run(result)#这里出结果
array([4, 7])
Tensorflow中的张量和Numpy里的数组有区别，TF计算的结果是一个张量结构，
保存了三个属性：名字(name)、维度(shape)、类型(type)
张量的命名就可以通过“ node:src_output ”的
形式来给出。其中node 为节点的名称， src一output 表示当前张量来自节点的第几个输出。
比如上面代码打出来的"add:0"就说明了result 这个张量是计算节点“ add ” 输出的第0个结果

会话
会话拥有并管理TensorFlow 程序运
行时的所有资源。所有计算完成之后需要关闭会话来帮助系统回收资源
sess = tf.Session()
sess.run(result)
sess.close()
使用上下文管理器
with tf.Session() as sess:
	sess.run(result)
array([2., 4.], dtype=float32)

sess = tf.Session()#指定sess为默认会话
with sess.as_default():
	print (result.eval())
[2. 4.]

tensorflow自动生成一个默认计算图,不指定计算图的运算会自动加入默认运算图
sess = tf.Session()
with sess.as_default():
	print(result.eval())
或者
sess = tf.Session()
print(sess.run(result))
print(result.eval(session=sess))

交互式环境里可以使用：
>>>sess = tf.InteractiveSession()#自动注册为默认会话
>>>print(result.eval())
>>>sess.close()
通过tf.InteractiveSession 函数可以省去将产生的会话注册为默认会话的过程

config = tf.ConfigProto(
		allow soft placement=True,#参数为自动调整在cpu和gpu上运行
		log_device_placement=True)#日志记录每个节点被安排在哪个设备上
#设置运行session的参数
#注册为默认会话，第一个参数为True可以自动调整运行在cpu或gpu上
sessl = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)

前向传播算法
之所以称之为全连接神经网络是因为相邻两层之间任意两个节点之间都有连接
矩阵乘法实现，x是输入，w1，w2是参数
a = tf.matmul(x,w1) #相当于numpy里的dot操作
y = tf.matmul(a,w2)

weights = tf.Variable(tf.random_normal([2,3], stddev=2))
随机生成函数
tf.random_normal	正态分布均值为0	参数：平均值，标准差，类型
tf.truncated_normal		正态分布，如果随机值偏离平均值2个标准差，重新随机
tf.random_uniform	均匀分布
tf.random_gamma

tf.zeros	tf.zeros([2, 3), int32) > ((0, 0, O] , [O , 0, O]]
tf.ones	tf.ones([2, 3] , int32) -> [[1 , 1 , 1) , [1 , 1 , 1))
tf.fill	tf.fill([2, 3), 9) -> ((9, 9, 9) , (9 , 9, 9))
tf.constant常数	tf.constant((1 , 2, 3)) -> (1 ,2,3)

b = tf.Variable(tf.zeros([3,]))
w2 = tf.Variable(weights.initialized_value() * 2.0)

tf里的参数Variable需要初始化才能输出,
此时需要
sess=tf.InteractiveSession()
sess.run(w2.initializer)#初始化此参数
print(sess.run(w2))#然后才能输出数值
全局初始化Variable
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(w2))
tf.Variable本身就是一个张量,需要初始化才能计算并输出
tf.global_variables()#获取当前计算图所有变量
tf.trainable_variables()#获取可学习变量

w1 = tf.Variable(tf.random_normal([2,3],stddev=1),name='w1')
w2= tf.Variable(tf.random_normal([2,3],stddev=1),name='w2')
w1 = tf.assign(w2)#更新w1的值为w2的,维度一样
w3 = tf.Variable(tf.random_normal([3,3],stddev=1),name='w3')
tf.assign(w1,w3)会shape不匹配的错,tf.assign(w1,w3,validate_shape=False)

参数的数据类型不能改变
参数的维度通过设置validata_shape=False
tf.assign(wl , w2 , validate shape=False)

运用placeholder动态提供输入数据
计算图是一种预先定义最后才进行计算的结构，这里我们先定义w1，w2，x，但都不赋值，在最后才注册并把x作为字典变量输入计算出结果
import tensorflow as tf
wl = tf.Variable(tf.random normal([2, 3], stddev=l))
w2 = tf.Variable(tf.random normal([3, 1], stddev=l))
＃定义placeholder 作为存放输入数据的地方。这里维度也不一定要定义。
＃但如果维度是确定的，那么给出维度可以降低出错的概率。
x = tf.placeholder(tf.float32,shape=(1,2),name="x-input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y, feed_dict={x: [[0.7,0.9]]}))#输入x的数值
batch输入同样，只要扩大x的维度即可shape=(3,2) feed dict={x : [[0 . 7 , 0 . 9), [0 . 1 , 0 . 4] , [0.5 , 0 . 8]]}))

反向传播
＃使用sigmoid 函数将y 转换为0 ～ 1 之间的数值。转换后y 代表预测是正样本的概率， 1-y 代表
＃预测是负样本的概率。
y = tf.sigmoid(y)
＃定义损失函数来刻画预测值与真实值得差距。
cross_entropy = tf.reduce_ mean(
y_ * tf.log(tf.clip_by_value(y , 1e-10, 1.0))
+(1-y)*tf.log(tf.clip_by_value (l-y , 1e-10, 1.0)))
#定义学习率，在第4 章中将更加具体的介绍学习率。
learning_rate = 0.001
#反向传播优化方法
train step =\
	tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#常用优化方法tf.train.GradientDescentOptimizer 、tf.train.AdamOptimizer 和
#tf.train.MomentumOptimizer
sess.run(train_step)#对所有在GraphKeys.TRAINABLE_VARIABLES集合中的变量进行优化

完整的神经网络
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.palceholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
y = tf.sigmoid(y)#使用sigmoid 函数将y 转换为0 ～ 1 之间的数值。转换后y 代表预测是正样本的概率
learningrate = 0.001

cross entropy= -tf.reduce_mean(
	y * tf.log(tf.clip by value(y, 1e-10, 1.0))
	+(1-y)*tf.log(tf.clip_by_value (l - y , 1e-10 , 1.0 )))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#通过随机数生成一个模拟数据集。
rdm = RandomState(l)#RandomState是一个预载的生成随机张量的类，通过实例化可以生成各种随机张量，比如rdm.rand(2,3),rdm.randint(2,2)
dataset_size = 128
X = rdm.rand(dataset_size, 2)#生成128行的输入数据
Y = [ [int(xl+x2 < 1)] for (xl , x2 ) in X]#生成验证数据
#先取x1+x2小于1的样本，记为1，意味着合格的零件，Y就是一个128长度[0,0,1,1....]的列表，记载着每个零件的合格与否

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)#初始化所有参数

STEPS = 5000
for i in range(STEPS):
#这里每次选取8个样本进行学习(0,8)(8,16)(16,24)....(120,128)
	start = (i * batch_size ) % dataset_size#从0开始
	end = min(start+batch_size , dataset_size )#从8开始

	sess.run(train_step,feed_dict={x: X[start:end], y: Y[start:end]})
	if i % 1000 == 0 #每隔一定计算loss
		total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
	    print("after %d iterations,loss is %g", (%i,%total_cross_entropy))
































































































