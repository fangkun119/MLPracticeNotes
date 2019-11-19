代码AutoEncoders.07.特征可视化

~~~python
# Visualizing the extracted features
# 用的是<05 AutoEncoders: 栈式(Stacked)自动编码器3 - 一次训练一个自动编码器(在单张图中)>训练出的模型

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

with tf.Session() as sess:
    saver.restore(sess, "./my_model_one_at_a_time.ckpt")  #加载模型
    weights1_val = weights1.eval()   #第一层模型的权重

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plot_image(weights1_val.T[i])

save_fig("extracted_features_plot")
plt.show() 
~~~

￼
从上面的可视化结果看到，前4个特征对应于小块特征，第5个特征在寻找