代码RNN.15 RNN 自然语言处理.seq2seq.机器翻译

~~~python
# 演示代码中用到的tensor操作
import tensorflow as tf
reset_graph()

Y = tf.placeholder(tf.int32, [None, 4]) 
Y_input = Y[:, :-1]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    sess.run(Y_input, feed_dict={Y: [[1,2,3,0],[4,5,6,0],[7,8,9,0],[10,11,12,0],[13,14,15,0]]})
    print(Y_input.get_shape())
    Y_unstack = tf.unstack(tf.transpose(Y_input))
    print(Y_unstack)
    print(Y_unstack[0].get_shape())
    print(Y_unstack[1].get_shape())
    print(Y_unstack[2].get_shape())
# 输出
(?, 3)
[<tf.Tensor 'unstack:0' shape=(?,) dtype=int32>, <tf.Tensor 'unstack:1' shape=(?,) dtype=int32>, <tf.Tensor 'unstack:2' shape=(?,) dtype=int32>]
(?,)
(?,)
(?,)


# 代码
# The basic_rnn_seq2seq() function creates a simple Encoder/Decoder model: 
# it  
#     first runs an RNN to encode encoder_inputs into a state vector,
#     then runs a decoder initialized with the last encoder state on decoder_inputs. 
# Encoder and decoder use the same RNN cell type but they don't share parameters.

import tensorflow as tf
reset_graph()

n_steps = 50					# 50个时间片 (把句子的单词序列建模成时间片）
n_neurons = 200				# 1层200个RNN神经元
n_layers = 3					# 3层神经元
num_encoder_symbols = 20000   	# 编码词表(英语)的词空间词数是20000
num_decoder_symbols = 20000   	# 解码词表(法语)的词空间词数是20000
embedding_size = 150 			# embedding向量是150维
learning_rate = 0.01			# 学习率是0.01

X = tf.placeholder(tf.int32, [None, n_steps]) 			# English sentences (None * 时间片数目)
Y = tf.placeholder(tf.int32, [None, n_steps]) 			# French translations (None * 时间片数目）
W = tf.placeholder(tf.float32, [None, n_steps - 1, 1])	# 权重矩阵 None * (时间片数目 - 1) * 1
Y_input 	= Y[:, :-1]								# Y_input：	每个样本Y值(法语序列)前n_steps - 1个词
Y_target 	= Y[:, 1:]								# Y_target：	每个样本Y值(法语序列)后n_steps - 1个词

encoder_inputs = tf.unstack(tf.transpose(X)) 		# list of 1D tensors： X转置(n_steps*None)，按第一根轴un-stack，生成n_steps个一维向量
decoder_inputs = tf.unstack(tf.transpose(Y_input)) 	# list of 1D tensors： Y_input转置((时间片数目-1)*None)，按第一根轴un-stack，生成n_steps-1个一维向量

lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]
cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

# Embedding RNN sequence-to-sequence model, 过程: 
# 1. 	英文embedding编码: Embeds encoder_inputs (例子中的英文) by a newly created embedding (of shape [num_encoder_symbols x input_size, 例子中就是20000 * len(encoder_inputs)]). 
# 2. 	映射到状态向量：It runs an RNN to encode embedded encoder_inputs into a state vector. 
# 3. 	法文embedding编码：It embeds decoder_inputs (例子中的法文) by another newly created embedding (of shape [num_decoder_symbols x input_size, 例子中就是20000 * len(decoder_inputs)]). 
# 4. 	英文状态向量，映射到法问embedding编码：It runs RNN decoder, initialized with the last encoder state, on embedded decoder_inputs.
# Returns
# outputs: 
#   A list of the same length as decoder_inputs of 2D Tensors. The output is of shape [batch_size x cell.output_size] 
#   when output_projection is not None (and represents the dense representation of predicted tokens). 
#   when output_projection is None， It is of shape [batch_size x num_decoder_symbols, 例子中就是batch_size * 20000] . 
# state: 
#   The state of each decoder cell in each time-step. 
#   This is a list with length len(decoder_inputs) -- one item for each time-step. It is a 2D Tensor of shape [batch_size x cell.state_size].
output_seqs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,			# A list of 1D int32 Tensors of shape [batch_size]
    decoder_inputs,			# A list of 1D int32 Tensors of shape [batch_size]
    cell,					# tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols,  	# 20000, number of symbols on the encoder side.
    num_decoder_symbols,	# 20000, number of symbols on the decoder side.
    embedding_size)			# 150, the length of the embedding vector for each symbol

logits = tf.transpose(tf.unstack(output_seqs), perm=[1, 0, 2])
logits_flat = tf.reshape(logits, [-1, num_decoder_symbols])
Y_target_flat = tf.reshape(Y_target, [-1])
W_flat = tf.reshape(W, [-1])
xentropy = W_flat * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_target_flat, logits=logits_flat)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
…

~~~
