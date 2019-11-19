代码AutoEncoders.02: 栈式(Stacked)自动编码器1-所有层一起训练

# Let's build a stacked Autoencoder with 3 hidden layers and 1 output layer (ie. 2 stacked Autoencoders). We will use ELU activation, He initialization and L2 regularization.

# Note: since the tf.layers.dense() function is incompatible with tf.contrib.layers.arg_scope() (which is used in the book), we now use python's functools.partial() function instead. It makes it easy to create a my_dense_layer() function that just calls tf.layers.dense() with the desired parameters automatically set (unless they are overridden when calling my_dense_layer()).

~~~python
# 用MNIST作为数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

# 模型数据流图
reset_graph()

from functools import partial

n_inputs = 28 * 28			#5层网络神经元的数量
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01		#学习率
l2_reg = 0.0001			#l2正则化超参数

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer() # He权重初始化，相当于lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg) # L2 正则化
my_dense_layer = partial(tf.layers.dense,			# 每一层神经元都是用如下默认参数:(1)全联接层 (2)ELu激活(输出层除外) (3)He初始化 (4) L2正则项
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1)			# 初始化各层, 输出层不使用激活函数(直接用logits与样本对照、用MSE计算重建损失值)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))			# 原始MSE重建损失值
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)  	# L2正则项
loss = tf.add_n([reconstruction_loss] + reg_losses)					# 加总的到最终的重建损失值

optimizer = tf.train.AdamOptimizer(learning_rate)		
training_op = optimizer.minimize(loss)

# 模型训练
init = tf.global_variables_initializer()
saver = tf.train.Saver() # not shown in the book

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
        saver.save(sess, "./my_model_all_layers.ckpt")  

# output
# 0 Train MSE: 0.0150667
# 19% Train MSE: 0.0164884
# 2 Train MSE: 0.0173757
# 3 Train MSE: 0.0168781
# 4 Train MSE: 0.0155875

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
save_fig("reconstruction_plot")
~~~

￼





