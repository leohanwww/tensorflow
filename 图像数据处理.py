图像数据处理

使用TFRecord格式统一存储输入数据


message Example {
	Features features = 1;
};

message Features {
	map<string, Feature> feature = 1;
};

message Feature{
	oneof kind {
		ByteList bytes_list = 1;
		FloatList float_list = 2;
		Int64List int64_list = 3;
		}
};

将数据存入TFRecord
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符串型的属性
def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#读取mnist数据
mnist = input_data.read_data_sets('/path', dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = '/path/to/output.tfrecords'
#创建一个writer来写TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    images_raw = images[index].tostring()#将每个图像转换成一个字符串
    #将一个样例转化为Example Protocol Buffer，并写入
    example = tf.train.Example(features=tf.train.Feature(feature={
        'pixels':_int_64_feature(pixels),
        'label':_int_64_feature(np.argmax(labels[index])),
        'images_raw':_bytes_features(images_raw)}
    ))
    writer.write(example.SerializerToString())#写入TFRecord文件
writer.close()


读取TFRecord
import tensorflow as tf

#创建一个reader来读取tfr文件
reader = tf.TFRecordReader()
#创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['/path/to/output.tfrecords'])
#从队列中读取一个样例
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(#解析单个样例函数
    serialized_example,
    features={
        'images_raw':tf.FixedLenFeature([],tf.string),#解析为一个tensor
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64),
    })

image= tf.decode_raw(features[’image_raw’], tf.uint8}#将字符串tensor解析成数组
label = tf.cast(features[’label’], tf.int32}
pixels = tf.cast(features[’pixels’], tf.int32}

sess = tf.Session()
#启动多线程处理输入数据
coord = tf.train.Coordinator(}
threads = tf.train.start_queue_runners(sess=sess, coord=coord}
#每次运行可以读取TFRecord 文件中的一个样例。当所有样例读完之后，在此样例中程序
#会再从头读取。
for i in range(10} :
	print(sess.run([image, label, pixels]))



图像编码处理
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('/path','rb').read()#读取原始图像为字符串
with tf.Session() as sess:#对图像进行解码，使用的是jpeg，还有png等
    img_data = tf.image.decode_jpeg(image_raw_data)#结果是一个tensor
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()

    encode_image = tf.image.encode_jpeg(img_data)#将tensor编码成jpeg并存入文件
    with tf.gfile.FastGFlie('/path/to/output','wb') as f:
        f.write(encode_image.eval())

调整图像大小
image_raw_data = tf.gfile.FastGFile('/path','rb').read()
image_data = tf.decode_jpeg(image_raw_data)#解码图像
image_data = tf.image.convert_image_dtype(image_raw_data,dtype=tf.float32)#转换格式为浮点
resized = tf.image.resize_images(image_data,[300,300],method=0)

croped = tf.image.resize_images_with_crop_or_pad(img_data,400,400)#截取指定大小图像，图像够大就截取，不够就在周围填充0
central = tf.image.central_crop(image_data,0.5)#根据比例截取

#图像各种翻转
fliped = tf.image.flip_up_down(image_data)
fliped = tf.image.random_flip_up_down(image_data)
fliped = tf.image.flip_left_right(image_data)
fliped = tf.image.random_flip_left_right(image_data)
transposed = tf.image.transpose_image(image_data)

adjusted = tf.image.adjust_brightness(img_data,-0.5)#调整亮度
adjust_brightness = tf.clip_by_value(adjusted, 0.0, 1.0)#把亮度限定在正确范围内
adjusted = tf.image.random_brightness(image, random_range)
adjust_brightness = tf.image.adjust_contrast(image_data, 5)#调整对比度
adjusted = tf.image.adjust_hue(img_data, 0.3)#调整色彩
adjusted = tf.image.adjust_saturation(img_data, 5)#调整饱和度
adjusted = tf.image.per_image_standardization(img_data)#调整数值为0,方差为1

图像加标注框
batched = tf.expand_dims(
    tf.image.convert_image_dtype(img_data,tf.float32),0)

boxes = tf.constant([0.05, 0.05, 0.9, 0.7],[0.35, 0.47, 0.5, 0.56])#同时添加两个标注框
#参数是相对位置，[y_min,x_min,y_max,x_max]
boxed = tf.image.draw_bounding_boxes(batched,boxed)

完整图像预处理

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#随机调整一张图像的色彩，定义不同顺序调整亮度、对比度、饱和度和色相，具体使用的顺序会影响学习
def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_saturation(image,lower=0.5,upper=0.5)
        image = tf.image.random_brightness(image,max_delta=32. / 255. )
        image = tf.random_hue(image,max_delta=0.2)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image,max_delta=32. / 255. )
        image = tf.image.random_saturation(image,lower=0.5,upper=0.5)
        image = tf.random_hue(image,max_delta=0.2)
    elif color_ordering == 2:
        #其他转换顺序
    return tf.clip_by_value(image, 0.0, 1.0)#把图片每个元素值规定在范围内

#预处理图片
def preprocess_for_train(image,height,width,bbox):
    if bbox is None:#标注框没定义的话就取全图片
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    if image.dtype != tf.float32#转换图像像素值类型
        image = tf.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)#随机截取图像
	
    distort_image = tf.slice(image, bbox_begin, bbox_size)
	distorted_image = tf.image.resize_images(distort_image,[height,width],method=np.randint(4))#调整图像大小为神经网络的输入大小
    distort_image = tf.image.random_flip_left_right(distort_image)#随机左右翻转图像
    distort_image = distort_color(distort_image,np.random.randint(2))#随机调整图像颜色
    return distort_image

image_raw_data = tf.gfile.FastGFile(path,'rb').read()
with tf.Session() as sess:
    image_data = tf.image.decode(image_raw_data)
    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    for i in range(6)
        result = preprocess_for_train(image_data,299,299,boxes)
        plt.imgshow(result.eval())
        plt.show()

多线程处理数据输入

队列,处理输入数据的框架
import tensorflow as tf

q = tf.FIFOQueue(2,'int32')#指定一个先进先出队列，可以保存两个元素
#RandomShufffleQueue是随机进出队列
init = q.enqueue_many(([0,10],))#使用函数初始化队列中的元素,元素的值为0和10
x = q.dequeue()#出队列
y = x + 1
q_inc = q.enqueue([y])#加入队列
with tf.Session() as tf:
    init.run()#初始化队列
    for i in range(5):
        v,_ = sess.run([x,q_inc])
        print v


多线程操作
coord = tf.train.Coordinator()#创建一个实例来协同多线程
threads = [
	threading.Thread(target=MyLoop, args=(cord, i , )) for i in range(5)]
for t in threads: t.start()
coord.join(threads)

def MyLoop(coord, worker_id):
	#使用tf.Coordinator 类提供的协同工具判断当前线程是否市要停止。
	while not coord. should_stop ():
		#随机停止所有的线程。
		if np.random.rand() < 0.1
			print ” Stoping from id: %d\n” worker_id,
			#coord.request_stop()函数来通知其他线程停止。
			coord.request_stop()
		else:
		#打印当前线程的Id
			print ” Working on id ： %d\n ” % worker_id,
		#暂停l秒
		time.sleep(l)

队列管理
queue = tf.FIFOQueue(100,"float")
enqueue_op = queue.enqueue([tf.random_normal([1])])#入队操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)#启动几个线程,每个线程运行enqueue_op操作
tf.train.add_queue_runner(qr)#加入tf计算图上指定集合
out_tensor = queue.dequeue()

with tf.Session() as sess:
	coord = tf.train.Coordinator()#协同启动进程
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	#必须明确使用此函数启动所有线程,进行入队操作以供后期调用
	for _ in range(3): print(sess.run(out_tensor)[0])
	coord.request_stop()
	coord.join(threads)


输入文件队列

num_shards = 2#总文件数
instances_per_shard = 2#每个文件多少数据
#把输入转换成TFRecord
for i in range(num_shards):
	filename = ('/path/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
	writer = tf.python_io.TFRecordWriter(filename)
	for j in range(instances_per_shard):#将数据封装成Example结构并写入TFRcecord文件
		example = tf.train.Example(features=tf.train.Features(feature={
		'i': _int64_feature(i),
		'j': _int64_feature(j)
		}
		))
	writer.write(example.SerializerToString())
writer.close()

读取执行
files = tf.train.match_filenames_once('/path/data.tfrecords-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)#创建输入队列
reader = tf.TFRecordReader()
_, serialized_example= reader.read(filename_queue)
features = tf.parse_single_example(
	serialized_example,
	features = {
		'i': tf.FixedLenFeature([], tf.int64),
		'j': tf.FixedLenFeature([], tf.int64),
		})
		
with tf.Session() as sess:
	tf.local_variables_initializer().run()
	print(sess.run(files))
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	for i in range(6):
		print(sess.run([features['i'], features['j']]))
	coord.request_stop()
	coord.join(threads)
	

组合训练数据
batch_size = 3
example,label = features[i], features[j]
capacity = 1000 + 3 * batch_size

example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size,
	capacity=capacity)
	#此函数将[example, label]整理成输入batch队列,自动管理队列出入
with tf.Session() as sess:
	tf.initializer_all_variables.run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	for i in range(2):
		cur_exapmle_batch, cur_label_batch = sess.run(
			[example_batch, label_batch])
		print(cur_exapmle_batch, cur_label_batch)
	
	coord.request_stop()
	coord.join(threads)



完整数据处理框架

import tensorflow as tf

files = tf.train.match_filenames_once('path/file_pattern-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)
#产生输入队列的函数

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)#读取序列
features = tf.parse_single_example(serialized_example,
	features = {
	'image': tf.FixedLenFeature([], tf.string),
	'label': tf.FixedLenFeature([], tf.int64),
	'height': tf.FixedLenFeature([], tf.int64),
	'weight': tf.FixedLenFeature([], tf.int64),
	'channels': tf.FixedLenFeature([], tf.int64),
	}
	)
image = features['image']
label = features['label']
height = features['height']
weight = features['weight']
channels = features['channels']

decode_image = tf.decode_raw(image, tf.uint8)#解码tensor
decode_image.set_shape([height,weight,channels])
#前面的预处理图片函数
image_size = 299
distort_image = preprocess_for_train(decode_image, image_size, image_size, None)

#整理成输入batch队列
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
	[distorted_image, label], batch_size=batch_size,
	capacity=capacity, min_after_dequeue=min_after_dequeue
	)

logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()
	coord = tf.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for i in(training_step):
		sess.run(train_step)
		
	coord.request_stop()
	coord.join()


数据集操作

input_data = [1, 2, 3, 5, 8]
dataset = tf.data.Dataset.from_tensor_slice(input_data)
iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()

dataset = tf.data.TextLineDataset(input_files)#从文本构建

#使用TFRecord需要定义parsr
def parser(record):
	features = tf.parse_single_example(
		record,
		features={
			'f1': tf.FixedLenFeature([],tf.int64)
			'f2': tf.FixedLenFeature([],tf.int64)
			})
	return features['f1'], features['f2']

input_files = '/path/to/TFRecordfile'
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)#对每条数据调用parser方法
iterator = dataset.make_one_shot_iterator()
f1, f2 = iterator.get_next()
with tf.Session as sess:
	for i in range(10):
		print(sess.run[f1, f2])

#使用placeholder需要初始化
iterator = dataset.make_initializer_iterator()
#从TFRecord文件创建数据集,具体文件路径是palceholder
f1, f2 = iterator.get_next()
with tf.Session() as sess:#需要初始化
	sess.run(iterator.initializer,
			feed_dict={input_files:['/path/to/TFRecordfile']})
	while True:
		try:
			sess.run([f1, f2])
		except tf.errors.OutOfRangeError:
			break



处理输入数据集

import tensorflow as tf

train_files = tf.train.match_filenames_once('/path')#这里读取TFRecord文件
test_files = tf.train.match_filenames_once('/path')

def parser(record):
	features = tf.parse_single_example(
		record,
		features={
			'image': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
			'height': tf.FixedLenFeature([], tf.int64),
			'weight': tf.FixedLenFeature([], tf.int64),
			'channels': tf.FixedLenFeature([], tf.int64),
			}
	)

	decode_image = tf.decode_raw(features['image'], tf.uint8)
	decode_image.set_shape([features['height'], features['weight'], features['channels']])
	label = features['label']
	return decode_image, label
	
image_size = 299
batch_size = 100
shuffle_buffer = 10000
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)#把数据用parser解析

#开始shuffle和batching操作
dataset = dataset.map(
	lambda image, label: (
		preprocess_for_train(image, image_size, image_size, None), label)
	)
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

NUM_EPOCH = 10
dataset = dataset.repeat(NUM_EPOCH)#数据集重复次数
iterator = dataset.make_initializer_iterator()
image_batch, label_batch = iterator.get_next()#获取batch数据

logit = inference(image_batch)
loss =calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

test_dataset = tf.data.TFRecord(test_files)
test_dataset = test_dataset.map(parser).map(
	lambda label: ( 
	tf.image.resize_images(image, [image_size, image_size], label))
	)
test_dataset = test_dataset.batch(batch_size)

test_iterator = test_dataset.make_initializer_iterator()#定义测试数据上的迭代器
test_image_batch, test_label_batch = test_iterator.get_next()

test_logit = inference(test_image_batch)
predictions = tf.argmax(test_logit, axis=-1, output_type=tf.int32)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer(),
			tf.loacl_variables_initializer())

	sess.run(iterator.initializer)
	
	while True:
		try:
			sess.run(train_step)
		except tf.errors.OutOfRangeError:
			break

	sess.run(test_iterator.initializer)
	test_results =[]
	test_labels = []
	while True:
		try:
			pred, label = sess.run([predictions, test_label_batch])
			test_results.extend(pred)
			test_labels.extend(label)
		except tf.errors.OutOfRangeError:
			break
			
correct = (float(y == y_) for (y, y_） in zip(test_results, test_labels)]
accuracy= sum(correct) / len(correct)
print (”Test accuracy is:”, accuracy)
















































































