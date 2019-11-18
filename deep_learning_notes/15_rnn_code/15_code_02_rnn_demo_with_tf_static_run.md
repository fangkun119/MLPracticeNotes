代码RNN.02 RNN原理演示：使用TF高层API static_run()展开RNN预测（实现基本预测、不含模型权重训练）

~~~python
# 权重矩阵随机生成（这个列子没有训练权重向量，只是演示RNN神经元如何输出）
# tf.contrib.rnn was partially moved to the core API in TensorFlow 1.2. Most of the *Cell and *Wrapper classes are now available in tf.nn.rnn_cell, and the tf.contrib.rnn.static_rnn() function is available as tf.nn.static_rnn().

# 输入特征为3维、神经元为5个
reset_graph()
n_inputs = 3
n_neurons = 5

# 两个mini batch的输入样本
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

# 定义一层RNN神经元
# This class is equivalent as tf.keras.layers.SimpleRNNCell, and will be replaced by that in Tensorflow 2.0
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0, Y1 = output_seqs

# 初始化全局变量，定义输入样本
init = tf.global_variables_initializer()
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])

# 将样本喂给模型，得到输出结果
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
~~~
