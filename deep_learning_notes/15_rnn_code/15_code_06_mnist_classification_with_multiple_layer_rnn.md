代码RNN.06 RNN例子：MNIST图像分类（多层RNN）

~~~python
# 说明：其实CNN比RNN更适合做图像分类，这里用MNST图像分类来作为例子，只是因为和前面章节的比较相似
reset_graph()

# MNIST是28*28像素，将其建模为28个时间片序列，每个时间片序列的特征数是28（特征中不包含上一个时间片的序列预测值）
n_steps = 28
n_inputs = 28
n_outputs = 10
learning_rate = 0.001

# X值：每个样本是28*28的像素数组； Y值是手写数字图片对应的数字标签（与前一个时间片的序列值没有关系）
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# 构建多层神经网络，3层每层100个神经元
n_neurons = 100
n_layers = 3
layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

# 数据流图其余部分
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32) # 多层神经元输出的预测值拼接在一起
states_concat = tf.concat(axis=1, values=states)   # 多层神经元输出的预测值拼接在一起
logits = tf.layers.dense(states_concat, n_outputs)  # 交给全联接层预测
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # 预测并计算交叉熵
loss = tf.reduce_mean(xentropy) #每个样本预测得到的交叉熵的均值作为损失值
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1) #预测结果
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #训练集准确率 

# 初始化全局变量
init = tf.global_variables_initializer() 
 # 训练模型
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

