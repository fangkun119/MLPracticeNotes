代码CNN.06 MNIST图片分类

Note: instead of using the fully_connected(), conv2d() and dropout() functions from the tensorflow.contrib.layers module (as in the book), 
we now use the dense(), conv2d() and dropout() functions (respectively) from the tf.layers module, which did not exist when this chapter was written. 

This is preferable because anything in contrib may change or be deleted without notice, while tf.layers is part of the official API. 
As you will see, the code is mostly the same.

For all these functions:
* the scope parameter was renamed to name, and the _fn suffix was removed in all the parameters that had it (for example the activation_fnparameter was renamed to activation).
The other main differences in tf.layers.dense() are:
* the weights parameter was renamed to kernel (and the weights variable is now named "kernel" rather than "weights"),
* the default activation is None instead of tf.nn.relu
The other main differences in tf.layers.conv2d() are:
* the num_outputs parameter was renamed to filters,
* the stride parameter was renamed to strides,
* the default activation is now None instead of tf.nn.relu.
The other main differences in tf.layers.dropout() are:
* it takes the dropout rate (rate) rather than the keep probability (keep_prob). Of course, rate == 1 - keep_prob,
* the is_training parameters was renamed to training.


~~~python
# 图片属性：28*28，单通道
height = 28
width = 28
channels = 1
# 输入神经元数量
n_inputs = height * width # 卷积层1：特征图32个、接受野3*3、步长1，零填充
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"
# 卷积层2：特征图64个、接受野3*3、步长2，零填充
conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"
# 池化层：特征图数量与卷积层相同
pool3_fmaps = conv2_fmaps
# 全联接层1：输出神经元数量
n_fc1 = 64
# 全联接层2（输出层）：神经元输出数量
n_outputs = 10

reset_graph()

# 输入
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X") 	# 输入图片占位符
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels]) 	# 输入图片拉平成1维
    y = tf.placeholder(tf.int32, shape=[None], name="y")			# 输入标签
 # 卷积层1、2
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

# 池化层
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

# 全联接层1
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

# 输出层
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

# 模型的数据流图
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

# 训练集准确行
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 初始化全局变量
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# warning: tf.examples.tutorials.mnist is deprecated. We will use tf.keras.datasets.mnist instead.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_mnist_model")

# output
# 0 Last batch accuracy: 0.99 Test accuracy: 0.9775
# 1 Last batch accuracy: 0.98 Test accuracy: 0.9841
# 2 Last batch accuracy: 0.98 Test accuracy: 0.979
# 3 Last batch accuracy: 0.99 Test accuracy: 0.9886
# 4 Last batch accuracy: 0.99 Test accuracy: 0.9883
# 5 Last batch accuracy: 1.0 Test accuracy: 0.9892
# 6 Last batch accuracy: 0.99 Test accuracy: 0.9891
# 7 Last batch accuracy: 1.0 Test accuracy: 0.9899
# 8 Last batch accuracy: 1.0 Test accuracy: 0.9871
# 9 Last batch accuracy: 1.0 Test accuracy: 0.989
~~~



