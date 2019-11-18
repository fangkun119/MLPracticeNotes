代码RNN.01 RNN原理演示：Manual 实现基本的RNN预测（不含权重训练）

~~~python
# 权重矩阵随机生成（这个列子没有训练权重向量，只是演示RNN神经元如何输出）
reset_graph()

# 定义数据流图 # 输入特征向量是3维
n_inputs = 3
# 单层5个RNN神经元
n_neurons = 5
# 两个mini batch输入
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])
# 样本特征权重矩阵(一列对应一个神经元，一行对应一个输入特征：3行5列），权重随机生成（这个列子没有训练权重向量，只是演示RNN神经元如何输出）
Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
# 反向连接权重矩阵(一列对应一个神经元，一行对应一个反向连接回来的输出特征：5行5列），权重随机生成（这个列子没有训练权重向量，只是演示RNN神经元如何输出）
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
# 偏置向量(1行5列，没列对应一个神经元）
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
# 计算Y值，将结果交给激活函数tf.tanh
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

# 初始化全局变量
init = tf.global_variables_initializer()

# 两个mini-batch的样本特征
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
# 运行数据流图并求值
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
# 打印模型输出
print(Y0_val)
# [[-0.0664006   0.9625767   0.68105793  0.7091854  -0.898216  ]
#  [ 0.9977755  -0.719789   -0.9965761   0.9673924  -0.9998972 ]
#  [ 0.99999774 -0.99898803 -0.9999989   0.9967762  -0.9999999 ]
#  [ 1.         -1.         -1.         -0.99818915  0.9995087 ]]
print(Y1_val)
# [[ 1.         -1.         -1.          0.4020025  -0.9999998 ]
#  [-0.12210419  0.62805265  0.9671843  -0.9937122  -0.2583937 ]
#  [ 0.9999983  -0.9999994  -0.9999975  -0.85943305 -0.9999881 ]
#  [ 0.99928284 -0.99999815 -0.9999058   0.9857963  -0.92205757]]
~~~
