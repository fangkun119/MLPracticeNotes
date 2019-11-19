代码AutoEncoders.09：借助降噪让模型学习特征

Note: the book uses tf.contrib.layers.dropout() rather than tf.layers.dropout() (which did not exist when this chapter was written). It is now preferable to use tf.layers.dropout(), because anything in the contrib module may change or be deleted without notice. The tf.layers.dropout() function is almost identical to the tf.contrib.layers.dropout() function, except for a few minor differences. Most importantly:
* you must specify the dropout rate (rate) rather than the keep probability (keep_prob), where rate is simply equal to 1 - keep_prob,
* the is_training parameter is renamed to training.

方法1: 使用高斯噪声

~~~python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
noise_level = 1.0

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_noisy = X + noise_level * tf.random_normal(tf.shape(X))   #添加噪声之后的样本

hidden1 = tf.layers.dense(X_noisy, n_hidden1, activation=tf.nn.relu, name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")   
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu,  name="hidden3") 
outputs = tf.layers.dense(hidden3, n_outputs, name="outputs") 

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  #MSE：噪声样本重建值 v.s 原始样本

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 10
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
        saver.save(sess, "./my_model_stacked_denoising_gaussian.ckpt")
# 输出
# 0 Train MSE: 0.0440489
# 1 Train MSE: 0.0432517
# 2 Train MSE: 0.042057
# 3 Train MSE: 0.0409477
# 4 Train MSE: 0.0402107
# 5 Train MSE: 0.0388787
# 6 Train MSE: 0.0391096
# 7 Train MSE: 0.0421885
# 8 Train MSE: 0.0398648
# 9 Train MSE: 0.0408181
~~~

方法2: 使用随机打断输入特征（dropout）

~~~python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01

dropout_rate = 0.3

training = tf.placeholder_with_default(False, shape=(), name='training')

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
X_drop = tf.layers.dropout(X, dropout_rate, training=training)   #添加噪声之后的样本
hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu, name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")  
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu,  name="hidden3")
outputs = tf.layers.dense(hidden3, n_outputs, name="outputs") 

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))   #MSE：噪声样本重建值 v.s 原始样本

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 10
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, training: True})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "./my_model_stacked_denoising_dropout.ckpt")
# 输出
# 0 Train MSE: 0.0296476
# 1 Train MSE: 0.0275545
# 2 Train MSE: 0.0250731
# 3 Train MSE: 0.0254317
# 4 Train MSE: 0.0249076
# 5 Train MSE: 0.0250501
# 6 Train MSE: 0.024483
# 7 Train MSE: 0.0251505
# 8 Train MSE: 0.0243836
# 9 Train MSE: 0.0242349

# 可视化：样本->噪声样本->编码解码器重建->重建样本
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
show_reconstructed_digits(X, outputs, "./my_model_stacked_denoising_dropout.ckpt")
~~~

￼
