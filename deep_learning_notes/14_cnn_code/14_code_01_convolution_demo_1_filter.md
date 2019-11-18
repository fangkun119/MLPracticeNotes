代码CNN.01 卷积原理演示1 - 过滤器

卷积核示例

~~~python
from sklearn.datasets import load_sample_image

# 加载两幅图片
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")

# 取其中一张图，转成灰度图
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis=2).astype(np.float32)
images = image_grayscale.reshape(1, height, width, 1)

# 两个卷积核
fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)  #7行*7列*1颜色通道*2个卷积核
fmap[:, 3, 0, 0] = 1  
fmap[3, :, 0, 1] = 1
plot_image(fmap[:, :, 0, 0])
plt.show()
plot_image(fmap[:, :, 0, 1])
plt.show()
~~~

￼


用卷积核处理图像

~~~python
# 定义数据流图
reset_graph()
X = tf.placeholder(tf.float32, shape=(None, height, width, 1))  #样本节点（占位符）	
feature_maps = tf.constant(fmap)  #卷积核（常量）
convolution = tf.nn.conv2d(X, feature_maps, strides=[1,1,1,1], padding="SAME") #卷积操作

# 执行数据流图，对chain这张图的灰度图做卷积操作
with tf.Session() as sess:
    output = convolution.eval(feed_dict={X: images})

# 卷积之前的原始图片，卷积之后的图片
# 原始图片
plot_image(images[0, :, :, 0])
save_fig("china_original", tight_layout=False)
plt.show()
# 使用第一个卷积核卷积之后的图片
plot_image(output[0, :, :, 0])
save_fig("china_vertical", tight_layout=False)
plt.show()
# 使用第二个卷积核卷积之后的图片
plot_image(output[0, :, :, 1])
save_fig("china_horizontal", tight_layout=False)
plt.show()
~~~

￼



