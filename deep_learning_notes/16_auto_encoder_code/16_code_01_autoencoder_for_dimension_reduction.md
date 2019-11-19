代码AutoEncoders.01: 降维

~~~python
# 构建3D数据集，共200个
import numpy.random as rnd

rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

# 正则化数据集的到200个样本，100用做训练集，100用做测试集
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

# 构建数据流图
# Note: instead of using the fully_connected() function from the tensorflow.contrib.layers module (as in the book), we now use the dense()function from the tf.layers module, which did not exist when this chapter was written. This is preferable because anything in contrib may change or be deleted without notice, while tf.layers is part of the official API. As you will see, the code is mostly the same.
# The main differences relevant to this chapter are:
# * the scope parameter was renamed to name, and the _fn suffix was removed in all the parameters that had it (for example the activation_fnparameter was renamed to activation).
# * the weights parameter was renamed to kernel and the weights variable is now named "kernel" rather than "weights",
# * the bias variable is now named "bias" rather than "biases",
# * the default activation is None instead of tf.nn.relu

import tensorflow as tf

reset_graph()

n_inputs = 3    			# 输入层维度是3
n_hidden = 2   			# 编码层维度是2
n_outputs = n_inputs	# 输出层维度是3、与输入一样

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

# 训练模型
init = tf.global_variables_initializer()

n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train})
    codings_val = codings.eval(feed_dict={X: X_test}) # 在这里计算中间隐藏层的编码

# 可视化编码结果
fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
save_fig("linear_autoencoder_pca_plot")
plt.show()
~~~



