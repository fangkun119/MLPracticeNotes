代码CNN.05 池化层

~~~python
import numpy as np
from sklearn.datasets import load_sample_images

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)

batch_size, height, width, channels = dataset.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 	# vertical line
filters[3, :, :, 1] = 1  # horizontal line

# ksize = [batch_size, height, width, channels] 
# strides = [批次步幅(用来跳过一些实例)=1，垂直步幅，水平步幅，通道步幅(用来跳过一些输入通道或者特征图)=1] 
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(output[0].astype(np.uint8))  # plot the output for the 1st image
plt.show()
~~~

￼

原始图片

~~~python
plot_color_image(dataset[0])
save_fig("china_original")
plt.show()
~~~

￼

池化以后的图片

~~~python
plot_color_image(output[0])
save_fig("china_max_pool")
plt.show()
~~~

￼

