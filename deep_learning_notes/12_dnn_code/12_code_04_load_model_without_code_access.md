代码DNN.04：加载完整的模型（在不能access原始模型训练代码的情况下)

~~~python
reset_graph()

# 知识点：加载模型meta，查看模型的数据流图
saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")
for op in tf.get_default_graph().get_operations():
    print(op.name)
from tensorflow_graph_in_jupyter import show_graph
    show_graph(tf.get_default_graph())

# 知识点：创建数据流图，根据数据流图的结构，创建对应的张量和操作节点
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")
~~~

另一种方法： 如果当初写模型的人，为了方便后人重用，做了如下操作

~~~python
for op in (X, y, accuracy, training_op):
	tf.add_to_collection("my_important_ops", op)
~~~

那么在这一步，也可以用另一种方式加载

~~~python
# 知识点：取出先前存入的collection
X, y, accuracy, training_op = tf.get_collection("my_important_ops")
with tf.Session() as sess:
    # 知识点：在会话中加载模型数据
    saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")
~~~
