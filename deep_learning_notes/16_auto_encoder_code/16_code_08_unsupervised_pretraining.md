代码AutoEncoders.08: Unsupervised pretraining
￼

~~~python
# Let's create a small neural network for MNIST classification

# 变量
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_outputs = 10

learning_rate = 0.01
l2_reg = 0.0005

activation 	= tf.nn.elu
regularizer 	= tf.contrib.layers.l2_regularizer(l2_reg)
initializer 		= tf.contrib.layers.variance_scaling_initializer()

X 	= tf.placeholder(tf.float32, shape=[None, n_inputs])
y 	= tf.placeholder(tf.int32, shape=[None])

weights1_init 	= initializer([n_inputs, n_hidden1])
weights2_init 	= initializer([n_hidden1, n_hidden2])
weights3_init 	= initializer([n_hidden2, n_outputs])

weights1 	= tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 	= tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 	= tf.Variable(weights3_init, dtype=tf.float32, name="weights3")

biases1 	= tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 	= tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 	= tf.Variable(tf.zeros(n_outputs), name="biases3")

# 数据流图
hidden1 	= activation(tf.matmul(X, weights1) + biases1)							# 隐藏层1
hidden2 	= activation(tf.matmul(hidden1, weights2) + biases2)						# 隐藏层2
logits 	= tf.matmul(hidden2, weights3) + biases3								# 输出层(计算logits)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)	# 输出层（交叉熵）
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)		# 三层的L2正则化损失值
loss = cross_entropy + reg_loss												# 正则化之后的损失值
optimizer = tf.train.AdamOptimizer(learning_rate)								# 优化器
training_op = optimizer.minimize(loss)											# 训练步骤，最小化损失值

correct 	= tf.nn.in_top_k(logits, y, 1)					# 预测正确的实例
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))	# 训练集准确率

#  初始化全局变量，以及预训练权重保存器
init = tf.global_variables_initializer()
pretrain_saver = tf.train.Saver([weights1, weights2, biases1, biases2]) 
saver = tf.train.Saver()

#  常规训练方法（不使用预训练特征）
n_epochs = 4
batch_size = 150
n_labeled_instances = 20000

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = rnd.permutation(n_labeled_instances)[:batch_size]
            X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end=" ")
        saver.save(sess, "./my_model_supervised.ckpt")
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy:", accuracy_val)
# 输出
# 0 Train accuracy: 0.973333 Test accuracy: 0.9334
# 1 Train accuracy: 0.98 Test accuracy: 0.936
# 2 Train accuracy: 0.973333 Test accuracy: 0.9382
# 3 Train accuracy: 0.986667 Test accuracy: 0.9494

# 使用预训练特征的方法
n_epochs = 4
batch_size = 150
n_labeled_instances = 20000

# training_op = optimizer.minimize(loss, var_list=[weights3, biases3])  #可选地，冻结低层的权重

with tf.Session() as sess:
    init.run()
    # 载入之前预训练的低层模型，
   # 模型生成见05 AutoEncoders: 栈式(Stacked)自动编码器3
   # 模型载入见上面的代码 pretrain_saver = tf.train.Saver([weights1, weights2, biases1, biases2]) 
    pretrain_saver.restore(sess, "./my_model_cache_frozen.ckpt")	
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = rnd.permutation(n_labeled_instances)[:batch_size]
            X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end="\t")
        saver.save(sess, "./my_model_supervised_pretrained.ckpt")
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy:", accuracy_val)
~~~

