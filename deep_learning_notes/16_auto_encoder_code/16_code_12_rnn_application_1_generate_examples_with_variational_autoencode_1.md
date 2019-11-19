代码AutoEncoders.12.应用：使用变分编码器(Variational Autoencoder)生成样本

~~~python
# 变量，模型超参数
reset_graph()
from functools import partial

n_inputs 		= 28 * 28		# 输入层神经元数目，对应MNIST图片像素
n_hidden1 	= 500		# hidden1神经元数目
n_hidden2 	= 500		# hidden2神经元数目
n_hidden3 	= 20  		# hidden3位于正中间，是编码层，20个神经元
n_hidden4 	= n_hidden2	# 对称
n_hidden5 	= n_hidden1	# 对称
n_outputs 	= n_inputs	# 对称

learning_rate 	= 0.001		# 学习率
initializer = tf.contrib.layers.variance_scaling_initializer()   								# HE模型权重初始化
my_dense_layer = partial(tf.layers.dense,  activation=tf.nn.elu, kernel_initializer=initializer)  	# 各层默认参数: 全联接层，ELu激活，He初始化

# 数据流图
X 				= tf.placeholder(tf.float32, [None, n_inputs])
hidden1 			= my_dense_layer(X, n_hidden1)
hidden2 			= my_dense_layer(hidden1, n_hidden2)
hidden3_mean 	= my_dense_layer(hidden2, n_hidden3, activation=None)		 # 平均编码
hidden3_sigma 	= my_dense_layer(hidden2, n_hidden3, activation=None)		 # 标准差编码 = 平均编码 * 高斯噪声
noise 			= tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32) # 高斯噪声
hidden3 			= hidden3_mean + hidden3_sigma * noise					 # 编码层 = 平均编码+标准差编码 = 平均编码+平均编码*高斯噪声
hidden4 			= my_dense_layer(hidden3, n_hidden4)
hidden5 			= my_dense_layer(hidden4, n_hidden5)
logits 			= my_dense_layer(hidden5, n_outputs, activation=None)
outputs 			= tf.sigmoid(logits)										# 值域转到[0,1]，用于代表灰度值，可视化重建样本


xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)		# 常规重建成本：交叉熵，推动自动编码器重建输入样本
reconstruction_loss = tf.reduce_sum(xentropy)								
eps = 1e-10 	# 防止传0值给tf.log(…)函数
latent_loss = 0.5 * tf.reduce_sum(		# 潜在损耗：编码的目标高斯分布，与实际分布的KL距离，使得编码器看起来像是从简单的高斯分布中采样
	  tf.square(hidden3_sigma) + tf.square(hidden3_mean) - 1 - tf.log(eps + tf.square(hidden3_sigma)))
loss = reconstruction_loss + latent_loss									

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 50
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
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
        saver.save(sess, "./my_model_variational.ckpt")
# 0 Train total loss: 32440.1 	Reconstruction loss: 25031.5 	Latent loss: 7408.61
# 1 Train total loss: 30017.4 		Reconstruction loss: 23093.3 	Latent loss: 6924.14
# 2 Train total loss: 23337.9 	Reconstruction loss: 20221.0 	Latent loss: 3116.88
~~~




