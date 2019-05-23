二元分类 binary_crossentropy 
tf.keras.losses.binary_crossentropy



多分类 categorical_crossentropy
ont_hot标签 tf.keras.losses.categorical_crossentropy
整数标签 sparse_categorical_crossentropy



数值回归 mean_squared_error
tf.losses.mean_squared_error          mse
tf.keras.losses.mean_squared_error


序列学习 connectionist temporal classification




表4-1　为模型选择正确的最后一层激活和损失函数
问题类型			最后一层激活	损失函数
二分类问题			sigmoid 		binary_crossentropy
多分类、单标签问题	softmax 		categorical_crossentropy
多分类、多标签问题	sigmoid 		binary_crossentropy
回归到任意值		无				mse
回归到0~1范围内的值	sigmoid 		mse 或binary_crossentropy