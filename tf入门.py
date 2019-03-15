tf入门

计算图
import tensorf low as tf
a = tf.constant([1.0, 2.0], name =”a ”)
b = tf.constant([2.0, 3.0], name =”b ”)
result = a + b
Tensor Flow 会自动将定义的计算转化为计算图上的节点。在TensorFlow 程序中，系统会自
动维护一个默认的计算图，通过tf.get_default_graph 函数可以获取当前默认的计算图。

g1 = tf.Graph()#持通过tf.Graph 函数来生成新的计算图。不同计算图上的张量和运算都不会共享
with g1.as_default():
	#在计算图g1中定义变量v并设置初始值为0
	v = tf.get_variable(
		"v", initializer=tf.zeros([1,]))

g2 = tf.Graph()
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


张量
可以简单理解为多维数组

import tensorflow as tf
# tf.constant 是一个计算，这个计算的结果为一个张量， 保存在变量a 中。
a= tf.constant([l . O, 2 . 0] , name =” a ”)
b = tf.constant( [2 . 0 , 3 . 0] , name = ” b ”)
result = tf.add( a , b , name=” add ”)
print result
>>> Tensor( "add:0"， shape=(2,), dtype=float32)
>>> tf.Session().run(result)#这里出结果
array([4, 7])
Tensorflow中的张量和Numpy里的数组有区别，TF计算的结果是一个张量结构，
保存了三个属性：名字(name)、维度(shape)、类型(type)
张量的命名就可以通过“ node:src_ output ”的
形式来给出。其中node 为节点的名称， src一output 表示当前张量来自节点的第几个输出。
比如上面代码打出来的"add:0"就说明了result 这个张量是计算节点“ add ” 输出的第0个结果

会话
会话拥有并管理TensorFlow 程序运
行时的所有资源。所有计算完成之后需要关闭会话来帮助系统回收资源
sess = tf.Session()
sess.run(...)
sess.close()
使用上下文管理器
with tf.Session() as sess:
	sess.run()

sess = tf.Session()#指定sess为默认会话
with sess.as_default():
	print (result.eval())

交互式环境里可以使用：
>>>sess = tf.InteractiveSession()#通过设置默认会话的方式来获取张量的取值
>>>print(result.eval())
>>>sess.close()
通过tf.InteractiveSession 函数可以省去将产生的会话注册为默认会话的过程

config = tf.ConfigProto(allow soft placement=True,log_device_placement=True)
#注册为默认会话，第一个参数为True可以自动调整运行在cpu或gpu上
sessl = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=co nfig)











































































































































































