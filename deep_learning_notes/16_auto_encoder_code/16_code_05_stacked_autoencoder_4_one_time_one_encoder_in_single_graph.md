代码AutoEncoders.05: 栈式(Stacked)自动编码器3 - 一次训练一个自动编码器(在单张图中)

~~~python
# Training one Autoencoder at a time in a single graph

# Another approach is to use a single graph. 
# To do this, we create the graph for the full Stacked Autoencoder, but then we also add operations to train each Autoencoder independently: 
#    phase 1 trains the bottom and top layer (ie. the first Autoencoder) 
#    phase 2 trains the two middle layers (ie. the second Autoencoder).


# 模型超参数、变量
reset_graph()

# 5层的神经元数量
n_inputs		= 28 * 28
n_hidden1 	= 300
n_hidden2 	= 150  # 表示编码的那层
n_hidden3 	= n_hidden1
n_outputs 	= n_inputs

# 超参数
learning_rate = 0.01
l2_reg = 0.0001
 # 变量
activation 	= tf.nn.elu									# ELu激活函数
regularizer 	= tf.contrib.layers.l2_regularizer(l2_reg)				# L2正则项
initializer 		= tf.contrib.layers.variance_scaling_initializer()		# He模型权重初始化

X = tf.placeholder(tf.float32, shape=[None, n_inputs])				# 样本占位符

weights1_init 	= initializer([n_inputs, n_hidden1])					# 用He初始化的4层模型的参数（输入层不需要，所以共4层）
weights2_init 	= initializer([n_hidden1, n_hidden2])
weights3_init 	= initializer([n_hidden2, n_hidden3])
weights4_init 	= initializer([n_hidden3, n_outputs])

weights1 		= tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 		= tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 		= tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
weights4 		= tf.Variable(weights4_init, dtype=tf.float32, name="weights4")

biases1 		= tf.Variable(tf.zeros(n_hidden1), name="biases1")	# 4层模型的偏置（输入层不需要，所以共4层）
biases2 		= tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 		= tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 		= tf.Variable(tf.zeros(n_outputs), name="biases4")

# 数据流图
hidden1 		= activation(tf.matmul(X, weights1) + biases1)
hidden2 		= activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 		= activation(tf.matmul(hidden2, weights3) + biases3)
outputs 		= tf.matmul(hidden3, weights4) + biases4
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate)

# 第一阶段：bypass hidden2 and hidden3，用原始样本X训练 hidden1 (weights1, biases1) 和output (weight4, bias4) 层
with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4   # hidden1 = activation(tf.matmul(X, weights1) + biases1)是隐藏层1的输出
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)   # 最小化的是第一阶段的loss值（不涉及hidden2, hidden3）

# 第二阶段：冻结hidden1和output层的模型参数，用hidden1的输出(hidden1)训练hidden2(weights2, biases2), hidden3(weights3, biases3)
with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))  # hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss 
    train_vars = [weights2, biases2, weights3, biases3]  #冻结hidden1和output层的模型参数
    phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars) # freeze hidden1

# 变量初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 模型训练
training_ops = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

with tf.Session() as sess:
    init.run()
    for phase in range(2):
        print("Training phase #{}".format(phase + 1))
        for epoch in range(n_epochs[phase]):
            n_batches = mnist.train.num_examples // batch_sizes[phase]
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
                sess.run(training_ops[phase], feed_dict={X: X_batch})
            loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_one_at_a_time.ckpt")
    loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
    print("Test MSE:", loss_test)

# Output
# Training phase #1
# 0 Train MSE: 0.00740679
# 1 Train MSE: 0.00782866
# 2 Train MSE: 0.00772802
# 3 Train MSE: 0.00740893
# Training phase #2
# 0 Train MSE: 0.295499
# 1 Train MSE: 0.00594454
# 2 Train MSE: 0.00310264
# 3 Train MSE: 0.00249803
# Test MSE: 0.00979144

# 另一种训练方法：因为hidden1在阶段2是冻结的，相同样本输出相同。缓存hidden1的输出，可以避免重新计算、加快训练速度
training_ops = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

with tf.Session() as sess:
    init.run()
    for phase in range(2):
        print("Training phase #{}".format(phase + 1))
        if phase == 1:  #第二阶段
            hidden1_cache = hidden1.eval(feed_dict={X: mnist.train.images})
        for epoch in range(n_epochs[phase]):
            n_batches = mnist.train.num_examples // batch_sizes[phase]
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                if phase == 1:  #第二阶段
                    indices = rnd.permutation(mnist.train.num_examples)
                    hidden1_batch = hidden1_cache[indices[:batch_sizes[phase]]]
                    feed_dict = {hidden1: hidden1_batch}
                    sess.run(training_ops[phase], feed_dict=feed_dict)
                else:
                    X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
                    feed_dict = {X: X_batch}
                    sess.run(training_ops[phase], feed_dict=feed_dict)
            loss_train = reconstruction_losses[phase].eval(feed_dict=feed_dict)
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_cache_frozen.ckpt")
    loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
    print("Test MSE:", loss_test)
~~~

