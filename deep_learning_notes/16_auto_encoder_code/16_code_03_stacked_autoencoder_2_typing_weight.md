代码AutoEncoders.03: 栈式(Stacked)自动编码器2-权重绑定(Tying weights)

~~~python
# It is common to tie the weights of the encoder and the decoder (weights_decoder = tf.transpose(weights_encoder)). 
# Unfortunately this makes it impossible (or very tricky) to use the tf.layers.dense() function, so we need to build the Autoencoder manually:

# 模型超参数
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0005

# 数据流图
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    # 只初始化2层的权重，另外两层会绑定到相同的变量上，因此不需要初始化
    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])

    # 权重绑定    
    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
    weights3 = tf.transpose(weights2, name="weights3")  	# tied weights
    weights4 = tf.transpose(weights1, name="weights4")   	# tied weights

    # 每一层的偏置不需要绑定
    biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
    biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
    biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
    biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

    # 构建数据流图
    hidden1 = activation(tf.matmul(X, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4

    # 重建误差 + L2正则项
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    reg_loss = regularizer(weights1) + regularizer(weights2)  # 因为权重绑定的缘故，正则项计算有所不同
    loss = reconstruction_loss + reg_loss

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
 # 模型训练
saver = tf.train.Saver()

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "./my_model_tying_weights.ckpt")

# 输出
0 Train MSE: 0.0150667
19% Train MSE: 0.0164884
2 Train MSE: 0.0173757
3 Train MSE: 0.0168781
4 Train MSE: 0.0155875

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
show_reconstructed_digits(X, outputs, "./my_model_tying_weights.ckpt")
~~~

￼
