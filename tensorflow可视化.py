tensorflow可视化

通过命名空间组织图
同一命名空间下的节点缩成一个点,顶层命名空间的节点才会显示

import tensorflow as tf

with tf.name_scope('a'):
    ttt = tf.Variable([1])
    print(ttt.name) #输出 a/variable:0

    qqq = tf.get_variable('sk', [1])#get_variable不受name_scope影响
    print(qqq.name) #输出 sk:0
	
with tf.variable_scope('qq'):
	rgrg = tf.get_variable('iop', [1])
	print(rgrg.name) #输出 qq/iop:0
	

with tf.name_scope('input1'):#一个命名空间在tensorboard图上显示为一个节点
	input1 = tf.constant([1,2,3], name='input1')
with tf.name_scope('input2'):
	input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

运行时写入记录文件,记录运行时每一个节点的时间\空间开销
import tensorflow as tf

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(train_steps):
        xs, ys = mnist.train.next_batch(batch_size)
        if i % 1000 == 0:#不是一直记录
            run_options = tf.RunOptions(#运行时需要记录的信息
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata = tf.RunMetadata()#运行时记录信息的proto
            _, loss_value, step = sess.run(
                [train_op, loss, global_steps], 
                feed_dict={x:xs, y_:ys},
                options=run_options, run_metadata=run_metadata)
            #将节点运行时的信息写入日志文件
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            print(step, loss_value)
        else:
            _, loss_value, step = sess.run(
                [train_op, loss, global_steps], 
                feed_dict={x:xs, y_:ys})
				

监控指标

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def variable_summaries(var, name):
    with tf.name_scope('summaries'):#监控信息放到一个命名空间下
		tf.summary.histogram(name, var)
		#此函数记录张量里元素取值,并会生成表,在tensorboard对应栏目下显示
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean/' + name, mean)#生成平均值信息日志
		#记录变量平均值信息的日志标签,记录到命名空间mean里,name后指定了变量
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev/' + name, stddev)
		
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope(weights):
			weights = tf.Variable(tf.truncate_normal(input_dim, output_dim), stddev=0.1)
			variable_summaries(weights, layer_name + '/weights')
			
		with tf.name_scope(biases):
			biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
			variable_summaries(biases, layer_name + '/biases')
			
		with tf.name_scope('Wx_plus_b'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			#记录输出节点在激活函数前的分布
			tf.summary.histogram(layer_name + '/preactivations', preactivate)
		activations = act(preactivate, name='activation')#激活函数
		#记录输出节点激活后的分布
		tf.summary.histogram(layer_name + '/activations', activations)
		return activations

def main():
    mnist = input_data.read_data_sets(path, one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    #将向量还原成图片,当成图片写入日志
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, y_))
            tf.summary.scalar('cross entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            cross_predition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            tf.acst(cross_predition, tf.float32)
        tf.summary.scalar('accuracy', accuracy)

    #整理所有日志,以后只需sess.run一次
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(path, sess.graph)
        tf.global_variables_initializer().run()

        for i in rang(train_step):
            xs, ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x:xs, y:ys})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

if __name__ == '__main__':
    tf.app.run()






















































































































