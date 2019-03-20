Tensoflow主要依赖包

Protocol Buffer
处理结构化数据工具
使用Protocol Buffer 时需要先定义数据的格式（ schema ）还原一个序
列化之后的数据将需要使用到这个定义好的数据格式
message user{#每一个message 代表了一类结构化的数据
	optional string name = 1 ;
	required int32 id = 2 ;
	repeated string email = 3 ;#可重复的
}
TensorFlow 中的数据基本都是通过Protocol Buffer来组织的

Bazel
自动化构建工具，用来编译
Bazel内含项目空间wokspace，可以解释为一个文件夹，包含了编译一个软件所需的源代码及输出编译结果
在一个项目空间内， Bazel 通过BUILD 文件来找到需要编译的目标
-rw-rw-r-- root root 208 BUILD
-rw-rw-r-- root root 48 hello_lib.py
-rw-rw-r-- root root 47 hello main.py
-rw-rw-r-- root root 0 WORKSPACE #外部依赖文件

#hello_lib.py
def print_hello_world():
	print("Hello World")

#hello_main.py
import hello_lib
hello_lib.print_hello_world()

在BULID文件中定义两个编译目标
py_library(#定义函数的文件编译为library以供调用
	name = "hello_lib",
	src = [
		"hello_lib.py",
	]
)

py_binary(#程序主入口编译为binary二进制
	name = "hello_main",
	src = [
		"hello_main.py",
	],
	deps = [
		":hello_lib",
	],
)


安装tensorflow
docker安装
docker run -it tensorflow/tensorflow:1.4.0

pip安装		
linux	找到安装包，安装
windows		pip install tensorflow

源码安装
安装依赖包	安装Bazel	安装tensorflow


>>> a = tf.constant([1.0,2.0],name='a')
>>> a
<tf.Tensor 'a:0' shape=(2,) dtype=float32>
>>> b = tf.constant([2.0,3.0],name='b')
>>> result = a + b
>>> result
<tf.Tensor 'add:0' shape=(2,) dtype=float32>
>>> tf.Session().run(result)
array([3., 5.], dtype=float32)
要输出相加得到的结果，不能简单地直接输出result ，而需要先生成一个会话（ session),
并通过这个会话（ session ）来计算结果。到此，就实现了一个非常简单的TensorFlow 模型











































































































