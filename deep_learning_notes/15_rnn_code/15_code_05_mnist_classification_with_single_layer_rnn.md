代码RNN.05 RNN例子：MNIST图像分类 （单层RNN）

说明：其实CNN比RNN更适合做图像分类，这里用MNST图像分类来作为例子，只是因为和前面章节的比较相似

Note: the book uses tensorflow.contrib.layers.fully_connected() rather than tf.layers.dense() (which did not exist when this chapter was written). It is now preferable to use tf.layers.dense(), because anything in the contrib module may change or be deleted without notice. The dense() function is almost identical to the fully_connected() function. The main differences relevant to this chapter are:
* several parameters are renamed: scope becomes name, activation_fn becomes activation (and similarly the _fn suffix is removed from other parameters such as normalizer_fn), weights_initializer becomes kernel_initializer, etc.
* the default activation is now None rather than tf.nn.relu.
tf.examples.tutorials.mnist is deprecated. We will use tf.keras.datasets.mnist instead.

~~~python
reset_graph()

# 宽*高分别建模成RNN步数、输入向量长度
n_steps = 28  
n_inputs = 28
# RNN层神经元个数
n_neurons = 150
# 输出层(全联接层)神经元个数
n_outputs = 10

learning_rate = 0.001
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)  #防止GPU内存溢出

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  #交叉熵作为损失函数
loss = tf.reduce_mean(xentropy)  #损失值
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# 训练集准确率
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 初始化全局变量
init = tf.global_variables_initializer()

# 加载样本
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# 乱序并返回mini-batch
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

X_test = X_test.reshape((-1, n_steps, n_inputs))

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()  #tf.global_variables_initializer()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))   #X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
~~~


