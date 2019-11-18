代码RNN.10 多层RNN

~~~python
reset_graph()

# 定义数据流图 ##################

# 输入特征
n_inputs = 2 	# 输入特征是两维（包括前一个时间片的输出，以及另外一个当前时间片的外部特征）
n_steps = 5    	# 使用5个时间片作为RNN的输入和输出
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # 输入特征占位符

# 多层RNN结构
n_neurons = 100	# 每层100个神经元
n_layers = 3		# 3层
layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
# 时间序列展开
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# 初始化全局变量
init = tf.global_variables_initializer()
 # 用随机数来模拟样本特征，来展开时间序列
X_batch = np.random.rand(2, n_steps, n_inputs)
X_batch.shape  #输出 (2样本数, 5时间片数量, 2特征数(例如可以建模为前一个时间片的序列值，外加当前时间片的外部特征值)
with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch})
outputs_val.shape #输出 (2, 5, 100/*神经元数量，如果想把这100个值映射为1个序列值，可以再加一个全联接层，用下一个时间片的Y值作为targe value来训练模型*/)
~~~

￼



