计算加速

import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1,2])
    b = tf.constant([3,4])
with tf.device('/gpu:1'): #把计算放到gpu上  
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

#自动--gpu无法处理的运算放到cpu上
sess = tf.Session(config=tf.ConfigProto(
		allow_soft_placement=True, log_device_placement=True))
		

#动态分配显存,使得一个gpu可以同时运行多任务
config = tf.ConfigProto()
config.gpu_options.allow_grouth = True



异步模式不同设备可能同时更新参数,从而使得参数更新过头,
可以使用同步模式,同时读取参数和随机数据-进行反向传播-计算所有随机数据梯度的平均值-更新

多GPU并行

#计算每一个变量梯度的平均值
def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, _ in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)
			grad.append(expanded_g)
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads
	
def main():
		with tf.Graph().as_defult, tf.device('/cpu:0'):
			x, y = get_input()
			regularizer = tf.contrib.layers.l2_regularizer(REGULARIZERAZTION_RETE)
			globel_step = tf.get_variable(
				'global_step', [], initializer=tf.constant_initializer(0),
				 trainable=False)
			learning_rate = tf.train.exponential_dacay(
				LEARNING_RATE_BASE,
				global_step,
				train_size / batch_size,
				LEARNING_RATE_DECAY)
			train_step = tf.train.GradientDescentOptimizer(learning_rate)
			tower_grads = []
			reuse_variables = False
			for i in range(GPUs):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('GPU_%d' % i) as scope:
						cur_loss = get_loss(
							x, y_, regularizer, scope, reuse_variables)
						reuse_variables = True
						grads = train_step.compute_gradients(cur_loss)
						tower_grads.append(grads)


分布式tensorflow

#创建本地tensorflow集群
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)

#生成两个任务的集群
c = tf.constant('hello from server1 !')
cluster = tf.train.ClusterSpec(
	{"local": ["localhost:8888", "localhost:8889"]})
#通过集群配置生成server,参数指定当前启动任务
server = tf.train.Server(cluster, job_name="local", task_index=0)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
sess.run(c)
server.join()

#第二个任务
c = tf.constant('hello from server2 !')
#集群中每个任务需要用相同配置
cluster = tf.train.ClusterSpec(
	{"local": ["localhost:8888", "localhost:8889"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
sess.run(c)
server.join()
启动第一个任务后会执行并等待第二个任务


计算图之间分布式








































































































