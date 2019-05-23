'''
k折法训练模型并在新数据上测试结果
'''

k = 4
num_validation_samples = len(data) // k
np.shuffle(data)

validation_score_list = []
for fold in range(k):
    # 验证数据从第一块开始取
    validation_data = data[fold * num_validation_samples:
    (fold + 1) * num_validation_samples]
    # 训练数据取验证数据前面的块和后面的块
    training_data = data[:fold * num_validation_samples] +
        data[(fold + 1) * num_validation_samples:]
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_score_list.append(validation_score)

average_score = np.average(validation_score_list)

# 在所有非测试数据上训练最终模型
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)



数据预处理

向量化
输入数据规范特征如下: 取值0-1之间\所有特征取值应该在大致范围内
x -= x.mean(axis=0)
x /= x.std(axis=0)
将数据缺失设为0

减小过拟合的方法:
机器学习根本问题是优化和泛化,优化(optimization)在训练数据上得到良好性能,
泛化(generalization)应用模型在前所未见的数据上

方法0:加大训练数据量

训练开始,优化和泛化相关,优化在降低损失的时候,泛化的损失也在降低,此时是欠拟合
(underfit)
训练一定轮数后,泛化不再提高,而是降低,开始过拟合,此时模型只能学习到和训练数据有关的模式,这种模式对新数据来说是不必要的
最优解决办法是获得更多训练数据
次优解决办法是调节模型允许存储的信息量,让模型只能记住几个模式,优化过程会集中学习最重要的模式
这样的方法是正则化(regularization)

方法一:
减小网络大小是很好的优化办法,让模型可学习的参数变少,在开始选择相对较少的层,然后逐渐增加层,同时在验证数据上检验

方法二:
模型越简单,指参数值分布的熵更少,或更少的模型,常见方法是让权重只能取较小的值,这个方法叫权重正则化(regularization)
L1 regularization 添加的成本与权重的绝对值成正比
L2 regularization 添加的成本与权重的平方成正比
model.add(Dense(16, kernel_regularizer=regularizer.l2(0.001),
				activation='relu', input_shape=(10000,)))
				
from keras import regularizer

regularizer.l1(0.001)
regularizer.l1_l2(l1=0.001, l2=0.001)

方法三:
dropout正则化
在训练过程中随机将该层的一些输出特征置为0
layer_out *= np.random.randint(0, high=2, size=layer_out.shape)
layer_out /= 0.5 #比例放大/0.5 == *2
model.add(layers.Dropout(0.5))



































