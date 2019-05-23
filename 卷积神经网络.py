卷积神经网络

每两层之间所有节点都是相连的，称为全连接层
使用全连接处理图像28X28X500+500=392500个参数，参数太多，计算缓慢，容易过拟合
我们可以将卷积层和池化层看成自动图像特征提取的过程。在特征提取完成之后，仍然需要使用全连接层来完成分类任务。
CIFAR-10图片大小为32x32x3
卷积层1	池化层1	卷积层2	池化层2	全连接层1		全连接层2		softmax		分类结果


卷积层特性一: 从卷积窗口,如(3, 3)学到的模式有平移不变性(translation invariant),在图像右下角覆盖窗口学到的模式,可以在任何地方识别这个模式,如左上角

卷积层特性二: 卷积层能学习到模式的空间层次结构(spatial hierarchies patterns),第一个卷积层学习到较小的局部模式(如边缘),第二个卷积层将学习由第一个卷积层特征组成的更大模式

包含两个空间轴(高度和宽度)和一个深度轴(通道轴)的3D张量,其卷积结果也叫特征图(feature map),对于深度轴大于1的彩色图像,卷积运算从特征图(只有一个通道)提取图块,对这些图块使用同样的卷积,生成输出特征图(宽度\高度\任意深度),输出深度是层的参数,不再是RGB那样代表颜色,而是代表过滤器(filter)
例子:卷积层输入(28, 28, 1), 输出(26, 26, 32),每个通道都是(26, 26),它是过滤器的对输入的响应图(response map),表示这个过滤器模式在输入中不同位置的响应,深度轴的每个维度都是以一个特征(或过滤器)




为了使得卷积前向传播矩阵大小不变,可以使用全0填充
使用全0填充 'SAME'
out.shape = celi(in / stride)
不使用全0填充 'VALID'
out.shape = (in - filter + 1) / stride

输入层维度是32 32 3 卷积层维度是5 5 16 卷积层参数个数是5*5*3*16+16=1216个
且卷积层的参数个数和图片的大小无关，它只和过滤器的尺寸、深度以及当前
层节点矩阵的深度有关。这使得卷积神经网络可以很好地扩展到更大的图像数据上。

#过滤器尺寸
weights = tf.get_variable(
    'weights',[5,5,3,16],
    initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable(
    'biases',[16],
    initializer=tf.constant_initializer(0.1))
input_tensor = [n,x,y,c]
conv = tf.nn.conv2d(input, weights,strides=[1,1,1,1],padding='SAME')
#strides=[1,1,1,1]最前最后两位为固定的1,中间两位是步幅
conv_bias = tf.nn.bias_add(conv, biases)#给每个节点加上偏置的函数，不能直接加
actived_conv2d = tf.nn.relu(conv_bias)

池化层
tf.nn.max_pool(actived_conv2d,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#ksize是过滤器大小,第一和最后一个数字为1,中间是过滤器大小
tf.nn.avg_pool(...)


LeNet-5模型
具有7层的卷积网络
# -*- coding: utf-8 -*-
import tensorflow as tf

input_tensor = tf.placeholder(tf.float32,
    [batch_size,
    mnist_inference.IMAGE_SIZE,
    mnist_inference.IMAGE_SIZE,
    mnist_inference.NUM_CHANNELS]
    name='x-input'
output = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')

reshaped_xs = np.reshape(xs, [BATCH_SIZE,
								mnist_inference.IMAGE_SIZE,
								mnist_inference.IMAGE_SIZE,
								mnist_inference.NUM_CHANNELS])

#神经网络参数
INPUT_NODE=784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一卷积层参数
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二卷积层参数
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层节点个数
FC_SIZE = 512

def inference(input_tensor,train,regularizer):

#第一卷积层，输入为28x28，输出为28x28x32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            'weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP]
            initializer=tf.truncated_normal_initializer(stddv=0.1))
        conv1_biases = tf.get_variable(
            'bias',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.conv2d(input_tensor,conv1_weights,stride=[1,1,1,1],padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

#第一池化层，输入为28x28，输出为14x14x32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],stride=[1,2,2,1],padding='SAME')

#第二卷积层，输入为14x14，输出为14x14x64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            'weight',[CONV2_SIZE,CONV2_SIZE,NUM_CHANNELS,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            'bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

    conv2 = tf.nn.conv2d(pool1,conv2_weights,stride=[1,1,1,1],padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

#第二池化层，输入为14x14，输出为7x7x64
    with tf.variable_scope('layer4-pool'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],stride=[1,2,2,1],padding='SAME')

#第五层是全连接层，把上一层7x7x64的tensor拉直成为一个向量
    pool_shape = pool2.get_shape().as_list()#这个方法从tensor获得shape
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]#7x7x64长度
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])#pool_shape[0]是batch_size


    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            'weight',[nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
#只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias',[FC_SIZE],
            initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.drop(fc1, 0.5)
#dropout层一般只在全连接层使用，dropout在训练时随机将部分节点的输出改为0，可以避免过拟合问题

#第六层全连接层，输入为长度512的向量，输出为长度10的向量
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            'weight',[FC_SIZE,NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias',[NUM_LABELS],
            initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights) + fc2_biases
    return logit



Inception-v3模型，同时卷积边长为1，3，5的过滤器，使用全0填充,所有输出矩阵都与输入大小一致,然后把结果拼接
利用tensorflow-slim简洁实现卷积层
slim = tf.contrib.slim#加载slim库
net = slim.conv2d(input,32,[3,3])#参数为输入节点、卷积层深度、过滤器尺寸
#设置默认参数取值,第一个参数是一个函数列表,列表中的函数使用第二个,第三个,第四个参数
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],  stride=1,padding='VALID'):
    #省略前面的推导过程，到最后一步推导
    .......
	.......
	net = 上一层的输出
    with tf.variable_scope('Mixed_7c'):#这是一个需要分别卷积然后叠加的大模块
		#第一条路径，过滤器边长为1，深度为320的卷积层
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 320, [1,1],scope='Conv2d_0a_1x1')
		#第二条路径，这个结构本身也是一个Inception结构
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
		#使用tf.concat函数将多个矩阵拼接,第一个参数3是指在第三个维度，也就是深度上进行拼接
            branch_1 = tf.concat(3,[
                slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3')
                slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0a_3x1'))
		#第三条路径
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.cov2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(3,[slim.cov2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')])

     #第四条路径
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net,384,[1,1],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')
     #最后组合
        net = tf.concat(3,[branch_0,branch_1,branch_2,branch_3])


迁移学习
用训练好的inception-v3模型,将输出层前面的层(瓶颈层)提取,再通过一个单层全连接层





































































































































