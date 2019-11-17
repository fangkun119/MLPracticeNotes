代码DNN.07：只加载想要重用的层（在可以获得原始模型训练代码的情况下）

~~~python
reset_graph()

n_inputs = 28 * 28  # MNIST

# 3层从旧模型文件中加载
n_hidden1 = 300 # reused
n_hidden2 = 50   # reused
n_hidden3 = 50   # reused
# 2层新建
n_hidden4 = 20   # new!
n_outputs = 10    # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# 知识点：因为可以access模型代码，因此可以用代码编写的方式来设置各层的结构
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")   # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
   hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
   logits     = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new!
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
~~~

另一种方式

~~~python
# 从GraphKeys中获取要加载的3层的结构
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]") # regular expression
restore_saver = tf.train.Saver(reuse_vars) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    # 加载
    restore_saver.restore(sess, "./my_model_final.ckpt")
    for epoch in range(n_epochs):  # not shown in the book
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size): # not shown
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})  # not shown
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})  # not shown
        print(epoch, "Validation accuracy:", accuracy_val)  # not shown
    save_path = saver.save(sess, "./my_new_model_final.ckpt")
~~~
