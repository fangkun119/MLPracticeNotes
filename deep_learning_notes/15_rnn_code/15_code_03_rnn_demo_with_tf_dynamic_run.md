代码RNN.03 RNN原理演示：使用TF高层API dynamic_run()展开RNN预测（实现基本预测、不含模型权重训练）

~~~python
# 权重矩阵随机生成（这个列子没有训练权重向量，只是演示RNN神经元如何输出）

# tf.contrib.rnn was partially moved to the core API in TensorFlow 1.2. Most of the *Cell and *Wrapper classes are now available in tf.nn.rnn_cell, and the tf.contrib.rnn.static_rnn() function is available as tf.nn.static_rnn().

# 输入特征为3维、神经元为5个、一共训练两个时间片
n_steps = 2
n_inputs = 3
n_neurons = 5

# 定义数据流图
reset_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# 初始化全局变量
init = tf.global_variables_initializer()

X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print(outputs_val)
# [[[-0.85115266  0.87358344  0.5802911   0.8954789  -0.0557505 ]
#  [-0.999996    0.99999577  0.9981815   1.          0.37679607]]
#
# [[-0.9983293   0.9992038   0.98071456  0.999985    0.25192663]
#  [-0.7081804  -0.0772338  -0.85227895  0.5845349  -0.78780943]]
#
# [[-0.9999827   0.99999535  0.9992863   1.          0.5159072 ]
#  [-0.9993956   0.9984095   0.83422637  0.99999976 -0.47325212]]
#
# [[ 0.87888587  0.07356028  0.97216916  0.9998546  -0.7351168 ]
#  [-0.9134514   0.3600957   0.7624866   0.99817705  0.80142   ]]]

show_graph(tf.get_default_graph())
~~~