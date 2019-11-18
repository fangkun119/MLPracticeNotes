代码CNN.02 卷积原理演示2 - 用自定义过滤器组装卷积核

~~~python
import numpy as np
from sklearn.datasets import load_sample_images

# 加载两张图片
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# 创建两个Filter
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  	# vertical line，第3列置1
filters[3, :, :, 1] = 1  	# horizontal line，第3行置1

# Create a graph with input X plus a convolutional layer applying the 2 filters
# strides = [批次步幅(用来跳过一些实例)=1，垂直步幅，水平步幅，通道步幅(用来跳过一些输入通道或者特征图)=1] 
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")  # 两个方向步幅都是2，零填充

# 计算卷积
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset}) 

# 输出图片1在卷积操作后的得到的特征图中的第2张：[0 (图片1）, :, :, 1（特征图2)]
plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.show()
~~~


~~~python
# 输出2张图片在卷积操作后得到的属于各自的2个特征图
for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plot_image(output[image_index, :, :, feature_map_index])
        plt.show()
~~~


实践中不是用自定义过滤器来编写卷机核，而是使用tensorflow lib提供好的2D卷积核









