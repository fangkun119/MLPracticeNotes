代码RNN.13 RNN LSTM长期记忆单元、窥视孔连接、GRU

~~~python
reset_graph()

n_neurons = 150
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)

n_steps 	 = 28 # 将MNIST图片的高度建模为RNN时间片数目
n_inputs 	 = 28 # 将NNIST图片的宽度建模为RNN输入特征数
n_outputs	 = 10
n_layers	 = 3
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# 基本LSTM使用tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons) 
# 窥视孔连接使用tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, use_peepholes=True)
# GRU使用tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons) for layer in range(n_layers)]


multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

states  # state[-1]是最后一层LSTMCell，state[-1][1]是最后一层的短期状态h=…
(LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(?, 150) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(?, 150) dtype=float32>),
 LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_5:0' shape=(?, 150) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_6:0' shape=(?, 150) dtype=float32>),
 LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_7:0' shape=(?, 150) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_8:0' shape=(?, 150) dtype=float32>))

top_layer_h_state #150个LSTM神经元的输出
<tf.Tensor 'rnn/while/Exit_8:0' shape=(?, 150) dtype=float32>

n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
~~~





