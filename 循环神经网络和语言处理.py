循环神经网络RNN,处理和预测序列数据,较多的是语言,翻译

同一时刻的输入有两部分,一部分为上一个时刻状态,另一部分输入为当前时刻的数值

(上一时刻循环体状态+当前输入)作为一个整体输入tensor dot weights 生成当前状态传递给下一层
当前状态 dot 另一组weights 作为当前层输出

具体代码可以用RNN层直接写出


LSTM长短时记忆网络
靠sigmod神经元和一个按位乘法的操作,这两个操作合在一起就是一个门操作,之所以叫做门是因为sigmoid作为激活函数的全连接层会输出一个0到1之间的值,输入是当前输入和上一个单元的第一层输出
遗忘门根据当前输入x和上一时刻输出h决定哪些信息被遗忘,f=sigmoid(w1x + w2h),取值在(0,1)之间,接近0的遗忘,接近1的保留
输入门根据x和h决定哪些信息加入到状态c生成新的状态c

#直接定义一个LSTM结构
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
#state是个包含两个tensor的tuple,包括了state.c和state.h
state = lstm.zero_state(batch_size, tf.float32)
loss = 0.0
num_steps = 100 #训练数据的序列长度
for i in range(num_steps):
	if i > 0: tf.get_variable_scope().reuse_variables()
	#因为LSTM自动保存了变量,直接调用保存的参数
	lstm_out, state = lstm(current_input, state)
	#lstm_out:输出门 state:用于输出给下一状态
	#(current_input):当前输入 (state):当前状态
	final_output = fully_connected(lstm_out)
	#将当前lstm的输出传入一个全连接层(sigmoid)得到的输出
	loss += calc_loss(final_output, except_output)#当前时刻输出的损失累积到总loss里
#keras里有更加高封装的LSTM层
	

双向循环和深层循环神经网络

每个时刻,输入同时向正向和反向提供,每个方向独立产生输出值和状态值(给链条的下一个单元使用)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell#基本lstm层

stack_lstm = tf.nn.rnn_cell.MulityRNNCell(#堆叠的多层的lstm结构
	[lstm_cell(lstm_size) for _ in range(number_of_layers)]
	
state = stacked_lstm.zero_state(batch_size, tf.float32)#初始state可以认为是0
for i in range(len(num_steps)):
	if i > 0: tf.get_variable_scope().reuse_variables()
	stack_lstm_out, state = lstm(current_input, state)
	final_output = fully_connected(stack_lstm_out)
	loss += calc_loss(final_output, except_output)
	
Dropout循环神经网络在单层LSTM前向中不适用dropout,而在多层LSTM中间使用dropout
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
stack_lstm = tf.nn.rnn_cell.MulityRNNCell(
	[tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size)) for _ in range(number_of_layers)]


语言处理


预测一个单词出现的概率是由此单词前n个单词决定的模型,是简单的n-gram模型
p(wn|w1,w2,....wn-1) 此概率通常由一个softmax层产生

#词汇表
word_labels = tf.constant([2, 0])#假设词汇表大小为3,有两个单词2和0
predict_logits = tf.constant([[2.0,-1.0,3.0],[1.0, 0.0, -0.5]])#预测结果
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
	labels=word_labels, logits=predict_logits)
sess.run(loss)

word_prob_smooth = tf.constant([[0.01, 0.01, 0 . 98], [0.98, 0.01, 0.01]])
loss= tf.nn.softmax_cross_entropy_with_logits(#此函数需要输入为一组概率值(0-1)
	labels=word_prob_smooth, logits=predict_logits)
#运行结果为［ 0. 37656265 , 0. 48936883]
sess.run(loss)

整理语言学习所需的PTB自然语言数据集
RAW_DATA = 'path/to/ptb.train.txt'
VOCAB_OUTPUT = 'ptb.vocab'
counter = collection.Counter()
with codes.open(RAW_DATA, 'r', 'utf-8') as f:
	for line in f:
		for word in line.strip().split():
			counter[word] += 1
sorted_word_to_cnt = sorted(counter.items(),
							key=itemgetters[1],
							reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]#这就是每个单词出现一次的一个列表
sorted_words = ["<eos>"]+ sorted_words#eos是句子结束符
	



































