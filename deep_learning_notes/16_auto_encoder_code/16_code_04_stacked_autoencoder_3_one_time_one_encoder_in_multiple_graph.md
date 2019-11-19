代码AutoEncoders.04: 栈式(Stacked)自动编码器3 - 一次训练一个自动编码器(在多张图中)

~~~python
# There are many ways to train one Autoencoder at a time. The first approach is to train each Autoencoder using a different graph, then we create the Stacked Autoencoder by simply initializing it with the weights and biases copied from these Autoencoders.
# Let's create a function that will train one autoencoder and return the transformed training set (i.e., the output of the hidden layer) and the model parameters.

# 重置环境
reset_graph()
from functools import partial

# 在单独的一个图中训练单层的自动编码机的函数
def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
                      learning_rate = 0.01, l2_reg = 0.0005, seed=42,
                      hidden_activation=tf.nn.elu,
                      output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():   
        tf.set_random_seed(seed)  
        n_inputs = X_train.shape[1]
        my_dense_layer = partial(  # 给后面创建的dense加默认参数
            	tf.layers.dense,
            	kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  #HE初始化
            	kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])  								# 输入层
        hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")		# 隐藏层
        outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")	# 输出层

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  			# 原始重建误差
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)	# L2正则项
        loss = tf.add_n([reconstruction_loss] + reg_losses)					# 加入正则项之后的最终重建误差

        optimizer = tf.train.AdamOptimizer(learning_rate) 					# 优化器
        training_op = optimizer.minimize(loss)								# 模型训练操作节点

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        # 模型训练
        init.run()
        for epoch in range(n_epochs):
            # 在一个epoch中的训练操作
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                indices = rnd.permutation(len(X_train))[:batch_size]
                X_batch = X_train[indices]
                sess.run(training_op, feed_dict={X: X_batch})
          # 打印模型损失值
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
	# 样本喂给隐藏层后的输出
        hidden_val = hidden.eval(feed_dict={X: X_train})
	# 获得自动编码解码机的参数
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]

# 训练
# Now let's train two Autoencoders. The first one is trained on the training data, and the second is trained on the previous Autoencoder's hidden layer output:
# 最外面的两层：不需要激活函数，用原始样本来训练
hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, n_neurons=300, n_epochs=4, batch_size=150, output_activation=None)
# 中间的两层：用外层模型的输出作为训练样本，内层使用了ReLU激活函数
_, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=4, batch_size=150)

# 0 Train MSE: 0.0185175
# 1 Train MSE: 0.0186825
# 2 Train MSE: 0.0184675
# 3 Train MSE: 0.0192315
# 0 Train MSE: 0.00423611
# 1 Train MSE: 0.00483268
# 2 Train MSE: 0.00466874
# 3 Train MSE: 0.0044039

# 拷贝前面训练的到的单层的编码解码机的模型参数，组装成栈式(Stacked)编码解码机
# Finally, we can create a Stacked Autoencoder by simply reusing the weights and biases from the Autoencoders we just trained:

reset_graph()
n_inputs = 28*28

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4

# 测试
def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

show_reconstructed_digits(X, outputs)
~~~

￼

