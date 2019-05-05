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
threads= tf.train.start_queue_runners(sess=sess, coord=coord}
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

import tensorflow as tf
#使用tf.QueueRunner启动多个线程操作同一个队列
queue = tf.FIFIQueue(100,'float')
enqueue_op = queue.enqueue([tf.random_normal([1])])#定义队列入队操作
qr = tf.train.QueueRunner(queue,[enqueue_op] * 5)#表示启动5个线程，每个线程进行enqueue_op操作
tf.train.add_queue_runner(qr)#第二个参数不写则将qr加入默认集合tf.GraphKeys.QUEUE_RUNNERS
out_tensor = queue.dequeue()#定义出队操作

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



















































