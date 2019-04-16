图像数据处理

将mnist数据转换为TFRecord

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
mnist = input_data.read_data_sets('/path',dtype=tf.uint8,one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
#训练数据的图像分辨率
pixels = images.shape[1]
num_examples = mnist.train.num_examples
#输出TFRecord文件地址
filename = '/path/to/output.tfrecords'
#创建一个writer来写tfr文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    #将图像转换成一个字符串
    images_raw = images[index].tostring()
    #将一个样例转化为Example Protocol Buffer，并将所有信息写入
    example = tf.train.Example(features=tf.train.Feature(feature={
        'pixels':_int_64_feature(pixels),
        'label':_int_64_feature(np.argmax(labels[index])),
        'images_raw':_bytes_features(images_raw)}
    ))
    write.write(example.SerializerToString())
writer.close()

#创建一个reader来读取tfr文件
reader = tf.TFRRecordReader()
#创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['/path/to/output.tfrecords'])
#从队列中读取一个样例
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'images_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64),
    })

image= tf . decode_raw(features [ ’ image_raw ’], tf.uint8}
label = tf . cast(features [’ label ’ ] , tf . int32}
pixels = tf.cast(features [’ pixels ’) , tf . int32}

sess = tf . Session(}
#启动多线程处理输入数据
coord = tf.train.Coordinator(}
threads= tf . train . start_queue_runners(sess=sess , coord=coord}
#每次运行可以读取TFRecord 文件中的一个样， 例。当所有样例者fl 读完之后，在此样例中程序
#会再从头读取。
for i iηrange(lO} :
print sess . run([image , label , pixels]}


图像编码处理
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('/path','r').read()#读取原始图像
with tf.Session() as sess:#对图像进行解码，使用的是jpeg，还有png等
    img_data = tf.image.decode_jpeg(image_raw_data)#结果是一个tensor
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()

    encode_image = tf.image.encode_jpeg(img_data)#将tensor编码成jpeg并存入文件
    with tf.gfile.GFlie('/path/to/output','wb') as f:
        f.write(encode_image.eval())

调整图像大小
image_raw_data = tf.gfile.FastGFile('/path','rb').read()#读取原始图像
image_data = tf.decode_jpeg(    )#解码图像
image_data = tf.image.convert_image_dtype(image_raw_data,dtype=tf.float32)#转换格式为浮点
resized = tf.image.resize_images(image_data,[300,300],method=0)

croped = tf.image.resize_images_with_crop_or_pad(img_data,3000,3000)#截取指定大小图像，图像够大就截取，不够就在周围填充0
central = tf.image.central_crop(image_data,0.5)#根据比例截取

tf.image.flip_up_down(image_data)#上下翻转

adjusted = tf.image.adjust_brightness(img_data,-0.5)#调整亮度

标注框

batched = tf.expand_dims(
    tf.image.convert_image_dtype(img_data,tf.float32),0)
#需要扩展为4维tensor，解码后的增加一维
boxes = tf.constant([0.05,0.10,0.25,0.58])
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
    distort_image = tf.image.resize_images(distort_image,[height,width],method=np.randint(4))#调整图像大小为神经网络的输入大小
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


队列
队列和变量类似，都是有状态的点，其他节点可以修改它们的状态
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

with tf.Session() as sess:
    coord = tf.train.Coordinator()#这是启动线程
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)#启动队列
    for i in range(3): print sess.run(out_tensor)[0]
    
    coord.request_stop()
    coord.join(threads)



























































