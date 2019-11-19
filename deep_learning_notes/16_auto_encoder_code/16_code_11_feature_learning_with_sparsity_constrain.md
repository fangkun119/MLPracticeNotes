代码AutoEncoders.11：借助稀疏性约束让模型学习特征

~~~python
## To speed up training, you can normalize the inputs between 0 and 1, and use the cross entropy instead of the MSE for the cost function: ##
# logits = tf.layers.dense(hidden1, n_outputs)
# outputs = tf.nn.sigmoid(logits)
# xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
# reconstruction_loss = tf.reduce_mean(xentropy)

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 1000  # sparse codings
n_outputs = n_inputs

def kl_divergence(p, q):
    # Kullback Leibler divergence
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))

learning_rate = 0.01
sparsity_target = 0.1	# 稀疏度目标
sparsity_weight = 0.2 	# 稀疏度损失值权重

X = tf.placeholder(tf.float32, shape=[None, n_inputs]) 			# 输入层
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid) 	# 隐藏层1，因为在最中间，因此也是编码层，用sigmoid是为了让值域变为[0，1]
outputs = tf.layers.dense(hidden1, n_outputs) 					# 输出层

hidden1_mean = tf.reduce_mean(hidden1, axis=0) 						# 隐藏层1输出均值
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))	# 稀疏度损失值(表示与目标稀疏度的差距)
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) 				# 重建损失值
loss = reconstruction_loss + sparsity_weight * sparsity_loss				# 总损失值

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 1000

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch}) 			# 训练模型参数
        reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss, sparsity_loss, loss], feed_dict={X: X_batch}) #损失值
        print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val, "\tSparsity loss:", sparsity_loss_val, "\tTotal loss:", loss_val)
        saver.save(sess, "./my_model_sparse.ckpt")

# 输出
# 0 Train MSE: 0.134832 	Sparsity loss: 0.421739 	Total loss: 0.21918
# 1 Train MSE: 0.0587859 	Sparsity loss: 0.0108979 	Total loss: 0.0609655
# 2 Train MSE: 0.053738 	Sparsity loss: 0.0201038 	Total loss: 0.0577588
# …

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
show_reconstructed_digits(X, outputs, "./my_model_sparse.ckpt")
~~~
