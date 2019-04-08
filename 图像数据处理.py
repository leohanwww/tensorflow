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
coord = tf.train . Coordinator(}
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
image_raw_data = tf.gfile.FastGFile('/path','r').read()#读取原始图像
image_data = tf.decode_jpeg(    )#解码图像
image_data = tf.image.convert_image_dtype(image_raw_data,dtype=tf.float32)
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




































































































