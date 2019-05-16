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