代码RNN.08 RNN时间序列（用OutputProjectionWrapper实现全联接层）

￼

Let's create the RNN. 
It will contain 100 recurrent neurons and we will unroll it over 20 time steps since each traiing instance will be 20 inputs long. 
Each input will contain only one feature (the value at that time). 
The targets are also sequences of 20 inputs, each containing a single value:

~~~python
# 重置
reset_graph()
n_steps = 20

# 用来构造特征的函数，每次调用，返回序列中的下一段20个
t_min, t_max = 0, 30
resolution = 0.1

# 上面图中的函数，给出横坐标数组，计算对应的纵坐标数组
def time_series(t):
	# t: batch_size行n_steps列的随机数、每一行n_steps个随机数值域在[t0, t0+0.2]之间，t0随column number增加而增加
	return t * np.sin(t) / 3 + 2 * np.sin(t*5)

# 用随机数的方法模拟一个mini-batch，每个batch的样本覆盖所有的时间序列区间
def next_batch(batch_size, n_steps):
	# t0: 	batch_size行1列的随机数、值域在[0, 28]
	t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
	# Ts: 	batch_size行n_steps列的随机数、每一行n_steps个随机数值域在[t0, t0+0.2]之间
	Ts = t0 + np.arange(0., n_steps + 1) * resolution
	# Ts相当于顶部图中的横坐标、ys相当于顶部图中的纵坐标
	ys = time_series(Ts)
	# return
	# x_batch: *个X值(其实*=batchsize)，每个样本是一个20行1列的多维数组，内容来自time_series返回多维数组去掉最后一列 
	# y_batch: *个Y值(其实*=batchsize)，每个样本是一个20行1列的多维数组，内容来自为time_series返回多维数组去掉第一列 
	return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
	# 例子：np.c_[X_batch[0], y_batch[0]] 返回x_batch, y_batch中的第一个样本
	# array([[-4.35508859, -3.41890916],
	# [-3.41890916, -2.85019143],
	# [-2.85019143, -2.71857372],
	# [-2.71857372, -2.97442641],
	# [-2.97442641, -3.46169864],
       # [-3.46169864, -3.95694633],
       # [-3.95694633, -4.22497921],
       # [-4.22497921, -4.07749388],
       # [-4.07749388, -3.42032817],
       # [-3.42032817, -2.27775625],
       # [-2.27775625, -0.78786458],
       # [-0.78786458,  0.82987194],
       # [ 0.82987194,  2.32688864],
       # [ 2.32688864,  3.48491434],
       # [ 3.48491434,  4.16791461],
       # [ 4.16791461,  4.35389289],
       # [ 4.35389289,  4.13877857],
       # [ 4.13877857,  3.71146565],
       # [ 3.71146565,  3.30612875],
       # [ 3.30612875,  3.14350607]])

# 数据流图
reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100

# 每个样本特征
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  	# [样本数, 时间片数量=20，时间片特征纬度=1 (只有一个特征即前一个时间片的Y值))
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])  	# [样本数, 时间片数量=20，时间片特征纬度=1 (当前时间片的Y值))

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

n_outputs = 1
learning_rate = 0.001

# 手动实现的全联接层（没有用OutputProjectionWrapper）
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])   #rnn_outputs格式为[None, n_steps=20, n_inputs=1],  format成[None,100] 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)  #stacked_output格式为[None, 100]
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])    #格式化成[Nont, n_steps=20, n_input=1]

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    
    saver.save(sess, "./my_time_series_model")
~~~





