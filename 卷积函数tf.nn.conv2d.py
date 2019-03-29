卷积函数tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

**input : ** 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true


卷积层输出大小计算
假设输入大小为(H,W)，滤波器大小为(FH, FW)，输出大小为
(OH,OW)，填充为P，步幅为S，输出公式见：
OH = ((H + 2P -FH) / S) + 1
OW = ((W + 2P -FW) / S) + 1

通用的卷积时padding 的选择

如卷积核宽高为3时 padding 选择1
如卷积核宽高为5时 padding 选择2
如卷积核宽高为7时 padding 选择3

1、如果padding = ‘VALID’
new_height = new_width = (W – F + 1) / S （结果向上取整）
也就是说，conv2d的VALID方式不会在原有输入的基础上添加新的像素（假定我们的输入是图片数据，因为只有图片才有像素），输出矩阵的大小直接按照公式计算即可。
2、如果padding = ‘SAME’
new_height = new_width = W / S （结果向上取整）#简便的卷积新大小计算法

池化层输出大小计算,F是池化层的核大小，S是池化层的步幅
PW = (W - F)/S + 1
PH = (H - F)/s + 1






























































































































