代码CNN.03 卷积原理演示3 - 使用TF Lib提供的卷积核

~~~python
reset_graph()

X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
# ksize = [batch_size, height, width, channels] 
conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding="SAME")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(conv, feed_dict={X: dataset})

# 以灰度图的形式输出第1张图片经过卷积核处理后，得到的第2个特征图
plt.imshow(output[0, :, :, 1], cmap="gray") 
plt.show()
~~~

￼
