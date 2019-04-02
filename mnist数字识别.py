数字识别

#加载MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集，设置保存文件夹
mnist = input_data.read_data_sets("d:/tensorflow",one_hot=True)

print("train data size:", mnist.train.num_examples)
55000
print("Validating data size:", mnist.validation.num_examples)
5000
print("Test data size:", mnist.test.num_examples)
10000
print("Example training data:", mnist.train.images[0])
[ 0 . 0 . 0 . 0 . 0.380  0.376 0. ）#数字代表颜色深度
print("Example training data label:", mnist.train.labels[0])
[ 0. 0. 0. 0. 0 . 0 . 0 . 1. 0 . 0]

tensorflow完整程序训练神经网络

import trnsorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784 #输入层的节点数
OUTPUT_NODE = 10 #输出层的节点数

LAYER1_NODE = 500 #一个隐藏层，有500节点
BATCH_SIZE= 100

LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减
REGULARIZATION = 0.0001 #描述模型复杂度的正则化在损失函数中的系数
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#辅助函数，给定所有参数，返回正向传播的结果，使用了滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	if avg_class = None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
		return tf.matmul(layer1, weight2) + biases2

	else:#使用滑动平均衰减率
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
	x = tf.placeholder(tf.float32, [None,INPUT_NODE],name='x-input')
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
	weights1 = tf.Variable(tf.truncated_normal(INPUT_NODE,LAYER1_NODE),stddev=0.1)
	biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
	weights2 = tf.Variable(tf.truncate_normal(LAYER1_NODE,OUTPUT_NODE),stddev=0.1)
	biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
	y = inference(x, None, weights1, biases1, weights2, biases2)

	global_step = tf.Variable(0, trainable=False)#训练轮数
	variable_averages = tf.train.ExponentiaMovingAverage(MOVING_AVERAGE_DECAY,global_step)#滑动平均类实例化
	
	variable_averages_op = variable_averages.apply(tf.train_variables())
	#在所有可训练参数上使用滑动平均
	average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)#前向计算

	#交叉嫡损失函数
	cross_entropy = tf.nn.sparse_softmax_with_logits(logits=y, labels=tf.argmax(y_,1))#1代表只在第一个维度，即每行取最大值的下标
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	#L2正则化损失函数
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RETE)
	#正则化损失
	regularization = regularizer(weights1) + regularizer(weights2)
	loss = cross_entropy_mean + regularization#总损失

	learning_rate = tf.train.exponential_decay(#指数衰减学习率
		LEARNING_RATE_BASE,#基础学习率
		global_step,#当前迭代轮数
		mnist.train.num_examples / BATCH_SIZE, #所有训练数据学完需要的迭代次数
		LEARNING_RATE_DECAY #学习衰减率


train_step = tf.train.GradientDecentOptimizer(learning_rate).minimize(loss,global_step=global_step)#优化损失函数
#每过一遍数据要通过反向传播更新参数值
#也要平滑计算参数
#train_op = tf.group(train_step,variable_averages_op)和下面一样效果
with tf.control_dependencies([train_step,variable_averages_op]):
	train_op = tf.no_op(name='train')

correct_prediction = tf.equal(tf.argmx(average_y,1), tf.argmax(y_, 1))
#检验滑动平均模型前向传播结果正确,结果是一个batch_size的一维数组
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#将布尔型转换为实数型，计算均值

with tf.Session() as sess:
	tf.global_variable_initializer().run
	validate_feed = {#验证数据
		x: mnist.validation.images,
		y_: mnist.validation.labels
	}

	test_feed = {
		x: mnist.test.images,
		y_: mnist.test.labels
	}

	for i in range(TRAINING_STEPS):
		if i % 1000 == 0
			validate_acc = sess.run(accuracy, feed_dict=validate_feed)
			print('after %d training steps, validation accuracy using average model is %g' % (i, validate_acc))
		#产生这一轮使用的一个batch的训练数据，并进行训练
		xs,ys = mnist.train.next_batch(BATCH_SIZE)
		sess.run(train_op, feed_dict={x:xs, y_:ys})
	#训练结束后，在测试数据是检测最终正确率
	test_acc = sess.run(accuracy, feed_dict=test_feed)
	print('arter %d training steps, test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))

def main():
	mnist = input_data.read_data_sets("d:/tensorflow", one_hot=True)
	train(mnist)

if __name__ == '__main__'
	tf.app.run()

变量管理

tf.Variable和tf.get_variable一样
tf.Variable(tf.constant(0.1,shape=[1]),name='v')#name不是必须
tf.get_variable("v",shape=[1],initializer=tf.constant_initilizer(1.0))#name必须

tf.get_variable里的参数
变量名称必填
initializer=tf.constant_initilizer 常量
			tf.random_normal_initializer 正态分布随机变量
			tf.truncate_normal_initializer 在标准差内的正态分布
			tf.random_uniform_initilizer 平均分布的随机值
			tf.uniform_unit_scaling_initilazer 
			tf.zeros_initilizer 全为0
			tf.ones.initilizer 全为1

with tf.variable_scope('foo'):#在命名空间里创建变量
	tf.get_variable('v',[1],initializer=tf.constant_initializer(1.0))
with tf.variable_scope('foo',reuse=True):#使用reuse只能获得已经存在变量，想新获得一个不存在的变量会报错
	tf.get_variable('v',[1])

v1 = tf.variable('v',[1])
print v1.name #v:0
with tf.variable_scope('foo')
	v2 = tf.get_variable('v',[1])
	print v2.name #foo/v:0
v3 = tf.get_variable('foo/v',[1])
print v3.name #/foo/v:0
v3 == v2 #True

def inference(input_tensor,reuse=False):
	
	with tf.variable_scope('layer1',reuse=reuse):#根据函数传进来的reuse决定是否创建新变量
		weights = tf.get_variable('weights',[INPUT_NODE,LAYER1_NODE],initializer=tf.truncate_normal_initializer(stddev=0.1))
		biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.truncate_normal_initializer(stddev=0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

	with tf.variable_scope('layer2',reuse=reuse):
		weights = tf.get_variable('weights',[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncate_normal_initializer(stddev=0.1))
		biases = tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.truncate_normal_initializer(0.0))
		layer2 = tf.matmul(layer1,weights)+biases
	return layer2

x = tf.placeholder(tf.float32,[NONE,INPUT_NODE],name='x-input')
y = inference(x)
#重新推导直接用新的参数调用
new_x = ......
y = inference(new_x,True)


保存模型
vl = tf.variable(tf.constant(1.0 , shape=[l]), name = ” vl ”)
v2 = tf .variable(tf.constant(2.0 , shape=[l]) , name= ” v2 ”)
result = vl + v2
init_op = tf.global_variable_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init_op)
	saver.save(sess,"/path/to/model/model.ckpt")
#加载模型
with tf.Session() as sess:
	sess.restore(sess,'/path/to/model/model.ckpt')
	print(sess.run(result))

v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')
# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})
# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})

'''
#直接加载图
saver = tf.train.import_meta_graph('/path/to/model/model.ckpt/model.ckpt.meta')
with tf.Session():
	saver.restore(sess,'/path/to/model/model.ckpt')
	print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
'''
#为了保存或加载部分变量
saver = tf.train.Saver([v1])
此时v2没有被加载，在运行初始化之前没有值
#声明新的变量名（不在已保存的模型中）
vl = tf.Variable(tf.constant(1.0 , shape=[l]) , name=”other-vl”)
v2 = tf.Variable(tf.constant(2.0 , shape=[l]) , name =”other-v2”)
#使用一个字典来重命名变量可以就可以加载原来的模型了。这个字典指定了
#原来名称为vl 的变量现在加载到变量vl 中（名称为other-v1 ），名称为v2 的变量加载到变量v2 中〈名称为other-v2 ）
saver= tf.train.Saver ({"vl":vl, "v2":v2})#此时保存的模型里v1变量是‘other-v1’的值

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name='v')
for variable in tf.global_variables():
	print variable.name
#V:0  没有滑动平均，输出一个变量

#使用滑动平均
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())
for variable in tf.global_variables():
	print variable.name
#输出v:0和v:/ExponentiaMovingAverage:0，后一个是影子变量

saver = tf.train.Saver()
with tf.Session() as sess:
	init_op = tf.global_variable_initializer()
	sess.run(init_op)
	
	sess.run(tf.assign(v,10))#给s一个新的值
	sess.run(maintain_average_op)
	saver.save(sess,'path/to/the/model/model/ckpt')#此时已经保存了v的两个值了
	print(sess.run([v, ema.average(v)]))
#输出[10.0, 0.99999905]

v = tf.Variable(0, dtype=tf.float32, name='v')
saver = tf.train.Saver({'v/ExponentiaMovingAverage':v})#变量重命名，将原来变量v的滑动平均值ema直接赋值给v
with tf.Session() as sess:
	saver.restore(sess, 'path/to/the/model/model/ckpt')
	print(sess.run(v))
#输出0.9999905，已经只有滑动平均值了，因为前面只使用了ema的值赋值给v

使用variable_to_restore()
import tensorflow as tf
v = tf.Variable(0, dtype=tf.float32 , name=”v ”)
ema = tf.train.ExponetialMovingAverage(0.99)
#通过使用variables to restore 函数可以直接生成上面代码中提供的字典
#｛ ” v/ExponentialMovingAverage ”： v ｝。
#会输出：
#{ ’ v/ExponentialMovingAverage ’: <tensorflow . Variable ’ v : 0 ’ shape=() dtype=float32 ref>}
#其中后面的Variable 类就代表了变量v 。
print ema.variables_to_restore()

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
	saver.restore(sess ,”/path/to/model/model.ckpt”)
	print sess.run(v) #输出0.099999905 ，即原来模型中变量v 的滑动平均值。

持久化数据格式
元图 元图是由MetaGraphDef Protocol Buffer 定义的
message MetaGraphDef {
	MetainfoDef meta_info_def = 1 ;
	
	GraphDef graph_def = 2 ;
	SaverDef saver_def = 3 ;
	map<string, CollectionDef> collection_def = 4 ;
	map<string, SignatureDef> signature_def = 5 ;
	repeated AssetFileDef asset_file_def = 6 ;
}

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v2')
result = v1 + v2
saver = tf.train.Saver()
saver.export_meta_graph('/path/to/model.ckpt.meda.json', as_text=True)
#将上一部分保存的model.ckpt.meta计算元图导出为json文件

#meta_info_def属性是通过MetalnfoDef 定义的，它记录了TensorFlow 计算图中的元数据以及TensorFlow程序中所有使用到的运算方法的信息。
message MetainfoDef {
	string meta_graph_version = 1;
	OpList stripped_op_list = 2;#记录所有计算方法，每个方法只出现一次
	google.protobuf.Any any_info = 3;
	repeated string tags = 4;
	string tensorflow_version = 5;
	string tensorflow_git_version = 6;
}

#上面stripped_op_list的方法中的一种
message OpDef {
	string name = l ;
	repeated ArgDef input arg = 2;
	repeated ArgDef output_arg = 3;
	repeated AttrDef attr = 4;
	
	OpDeprecation deprecation = 8;
	string summary = 5;
	string description = 6;
	bool is commutative = 18;
	bo 。l is aggregate = 16
	bool is stateful = 17;
	bool allows_u 口initialized_input = 19;
}
#OpDef 类型中前4 个属性定义了一个运算最核心的信息。Op Def 中的第一个属性name定义了运算的名称，这也是一个运算唯一的标识符。

#流程：是一个嵌套的定义message
message MetaGraphDef(meta_info_def)	
	message MetaInfoDef(OpList stripped_op_list)
		message OpDef(string name)
			op(add)

op {
	name: "Add"
	input_arg {
		name: "X"
		type_attr: "T"
	}
	input_arg {
		name: "y"
		type_attr: "T"
	}
	
	output_arg {
		name: "z"
		type_attr: "T"
	}
	attr {
		name: "T"
		typr: "type"
		allowed_values {
			list {
				type: DT_HALF
				type: DT_FLOAT
				}
			}
		}
	}

#graph_def记录计算图上的节点信息，graph_def属性只关注运算的连接结构
message GraphDef {
	repeated NodeDef node = 1;
	VersionDef version = 4;
}
message NodeDef {#存储节点主要信息
	string name = 1;#节点唯一标识符
	string op = 2;#运算方法
	repeated string input = 3;#字符串列表，定义输入
	string devive = 4;#定义运算设备
	map<string, AttrValue> attr = 5;
};

import tensorflow as tf
reader = tf.NewCheckpointReader('/path/to/model/model.ckpt')
#读取checkpoint文件中保存的所有变量
global_variables = reader.get_variable_to_shape_map()
for variable in global_variables:
	print(variable_name, global_variables[variable_name])
print"value for variable v1 is", reader.get_tensor("v1")
#输出：
v1[1]
v2[1]
Value for variable vl is [l.] #变量vl的取值为1。






