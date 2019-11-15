# 使用Tensorflow

## 1. 库和工具
**skflow**: https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html<br/>
**TF-Slim (tensorflow.contrib.slim)**：简化神经网络的构建、训练和评估<br/>
**Keras (http://keras.io)**<br/>
**Preffy Tensor (https//github.com/goolge/prettytensor/)**：高级API<br/>
自动微分（autodiff）<br/>
TensorBoard：可视化工具，浏览计算图和学习曲线<br/>
**TF.Learn (tensorflow.contrib.learn，不是TFLearn库，TFLearn库是一个独立的项目)**：兼容scikit-Learn<br/>

~~~python
import tf.contrib.learn.DNNClassifier
import tf.contrib.learn.DNNRegressor
import tf.contrib.learn.DNNLinearCombineRegressor
import tf.contrib.learn.DNNLinearCombinedClassifier
import tf.contrib.learn.LinearClassifier
import tf.contrib.learn.LinearRegressor
import tf.contrib.learn.LogisticRegressor
…
~~~ 
contrib下面的库不稳定，代码会被deprecate

## 2.服务
云服务： http://cloud.google.com/ml/ <br/>
社区： <br/>
>	https://www.tensorlfow.org  <br/>
>	https://github.com/jtoy/awesom-tensorflow <br/>

答疑： <br/>
>	http://stackoverflow.com/ 并将问题tag为tensorflow

谷歌讨论组： http://goo.gl/N7kRF9 

## 3.安装：
P207

## 4.计算流图与会话
**(1) 定义计算流图**
> P208顶部

**(2) 在会话中执行计算流图**

~~~python
sess = tf.Session()
sess.run(x.initializer)   	# x = tf.Variable(3, name=“x”)
sess.run(y.initializer)   	# y = tf.Variable(4, name=“y”)
result = sess.run(f)      	# f = x * x * y + y + 2
print(result)
sess.close() 			# 需要显示地关闭session
~~~
**(3) 在会话中执行计算流图（更简洁的写法），同时session会在离开with块时自动关闭**

~~~python
with tf.Session() as sess:
	x.initializer.run() 
	y.initializer.run()
	result = f.eval()
	# session自动关闭
~~~

**(4) 更简单的写法（创建一个全局变量初始化节点，该节点运行回自动调用全局变量的初始化，不需要手动运行`*.initializer.run()`**

~~~python
init = tf.global_variables_initializer() 
with tf.Session() as sess:
	init.run()
	result = f.eval() 
	# session自动关闭
~~~

**(5) InteractiveSession会在创建时自动被设置为默认Session，不需要with块**
P209

### 6.图管理
**(1) 默认情况下节点自动添加到default graph上**

~~~python
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph() #输出true
~~~

**(2) 管理多个数据流图时，可以将其中一个临时设置为default graph**

~~~python
graph = tf.Graph()
with graph.as_default():
       x2 = tf.Variable(2)
~~~

### 7.节点、变量
**(1) 生命周期**
节点的生命周期是数据流图的每次执行；变量的生命周期伴随整个session从打开到关闭

**(2) 对节点求值，其依赖的节点会跟着一起被递归求值，但求的的值不会被复用，执行结束后就关闭了**

~~~python
w = tf.constant(3)
x = w + 1
y = x + 1
z = x + 2
with tf.Session() as sess:
	print(y.eval()) #对y, x, w求值
	print(z.eval()) #对z, x, w求值
 	#x,w被求值了两次
~~~

**(3) 如果希望节点求值结果被复用，需要显示地保存**

~~~python
with tf.Session() as sess:
	y_val, z_val = sess.run([y,z])
	print(y_val)
	print(z_val)
~~~

**(4) 变量的生命周期伴随整个session从打开到关闭，在session执行期间可以被复用**

* 同名变量在不同Session中拥有各自的副本、相互独立
* 分布式TensorFlow中即使变量共享同一个计算图，多个会话之间相互隔离，不共享任何状态

### 8.例子：用公式求解法在Tensorflow中做线性回归：
完整内容：P211
因为使用的是tensorFlow的API，诸如：

~~~python
tf.tansapose(X)
tf.matmul(XT,X)
tf.magrix_inverse(Z)
~~~
如果有GPU，这些运算会放在GPU上进行

### 9.例子：手动计算线性回归中的梯度下降
**(1) 注意事项**：需要先做向量归一化否则会很慢<br/>
**(2) API**:

~~~python
tf.constant(…, dtype=xxx, name="xxx")
tf.Variable(…, name="xxx")
tf.reduce_mean(tf.square(error_arr), name="mse")
tf.transpose(X)
tf.assign(xxx, yyy)
tf.global_variables_initializer()
~~~

**(3) 代码**:

~~~python
reset_graph()
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
~~~
X, Y定义成常量；theta定义成Variable；其他是普通节点包括y<sub>pred</sub>, error, mse以及 gradiet；training\_op是一个tf.assign节点，用于计算和更新Vairable θ的值

### 10. 自动微分，以上面的线性回归为例

**(1) 知识**</br> 用偏导公式计算梯度向量：<br/>

~~~python
gradients = 2/m * tf.matmul(tf.transpose(X), error)
~~~
 用自动微分计算梯度向量：<br/>

~~~python
gradients = tf.gradients(mse, [theta])[0]  #返回mse相对于theta的梯度向量，mse是每个样本预测y值的error组成的向量
~~~
 自动计算梯度的方法有4种：数值微分，符号微分，前向自动微分，后向自动微分 （原理见附录D） tensorflow使用的是反向自动微分，适合有多个输入和少量输出的场景，计算效率高，遍历N_output + 1次就可以求出所有输出相对于输入的偏导

(2) 代码<br/>

~~~python
reset_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
print("Best theta:")
print(best_theta)
~~~

### 11.使用梯度下降优化器
方法：<br/>
直接调用自动微分：<br/>

~~~python
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)
~~~

使用梯度下降优化器

~~~python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = otptimizer.minimize(mse) 
~~~

代码

~~~python
reset_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
print("Best theta:")
print(best_theta)
~~~

### 12.动量优化器
特点：比梯度下降收敛更快 <br/>
方法：<br/>
使用梯度下降优化器<br/>

~~~python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
~~~

使用动量优化器<br/>

~~~python
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
~~~


代码：<br/>

~~~python
reset_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        sess.run(training_op)
    best_theta = theta.eval()
print("Best theta:")
print(best_theta)
~~~

### 13.用PlaceHolder为训练算法提供数据<br/>

~~~python
reset_graph()
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)
~~~

### 14.实现Mini-Batch梯度下降
方法：借助placeHolder，见下面代码加粗部分<br/>

~~~python 
reset_graph()
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
    print(best_theta)
~~~

### 15. 保存和加载模型，可以保存训练过程中的快照，也可以保存训练后的最终结果

~~~python
reset_graph()
n_epochs = 1000                                                                  
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X") 
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y") 
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions") 
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse") 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  
training_op = optimizer.minimize(mse) 

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval()) 
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
print(best_theta)

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval() # not shown in the book
np.allclose(best_theta, best_theta_restored)

saver = tf.train.Saver({"weights": theta})
reset_graph(). # notice that we start with an empty graph.

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book
~~~

### 18. 在JupyterNotebook中用TensorBoard可视化当前的数据流图

~~~python
# To visualize the graph within Jupyter, we will use a TensorBoard server available online at https://tensorboard.appspot.com/ (so this will not work if you do not have Internet access). As far as I can tell, this code was originally written by Alex Mordvintsev in his DeepDream tutorial. Alternatively, you could use a tool like tfgraphviz.
from tensorflow_graph_in_jupyter import show_graph
show_graph(tf.get_default_graph())
~~~

### 17.用TensorBoard来可视化数据流图和学习曲线

**方法**：<br/> 
创建FileWriter，将需要可视化的数据写到File Writer中，然后运行tensor board程序来可视化： 

**代码**：<br/> 

~~~python
reset_graph()
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y  = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess: 
    sess.run(init)

    for epoch in range(n_epochs):   
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval() 

file_writer.close()
print (best_theta)
~~~

### 19.命名作用域，也会直接影响到变量命名

~~~python
reset_graph()

a1 = tf.Variable(0, name="a")      # name == "a"
a2 = tf.Variable(0, name="a")      # name == "a_1"

with tf.name_scope("param"):       # name == "param"
    a3 = tf.Variable(0, name="a")  # name == "param/a"

with tf.name_scope("param"):       # name == "param_1"
    a4 = tf.Variable(0, name="a")  # name == "param_1/a"

for node in (a1, a2, a3, a4):
    print(node.op.name)
~~~

### 20.带有命名作用域的模块化

~~~python
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1) 
        w = tf.Variable(tf.random_normal(w_shape), name="weights") 
        b = tf.Variable(0.0, name="bias") 
        z = tf.add(tf.matmul(X, w), b, name="z")  
        return tf.maximum(z, 0., name="max") 

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
file_writer.close()
~~~

### 21. 在图的不同组件中共享变量
经典方法1: 函数传参

~~~python
reset_graph()

def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1) 
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  
        b = tf.Variable(0.0, name="bias") 
        z = tf.add(tf.matmul(X, w), b, name="z") 
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")
~~~

经典方法2: 构建一个通用字典，供每一个函数（用命名作用域来控制）使用

~~~python
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1 
        w = tf.Variable(tf.random_normal(w_shape), name="weights") 
        b = tf.Variable(0.0, name="bias")  
        z = tf.add(tf.matmul(X, w), b, name="z") 
        return tf.maximum(z, relu.threshold, name="max")
~~~

推荐方法3: 使用getVaraible (版本1:必须在外部定义）

~~~python
reset_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):  #表示是使用relu Variable，而不是定义它
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1 
        w = tf.Variable(tf.random_normal(w_shape), name="weights") 
        b = tf.Variable(0.0, name="bias")   
        z = tf.add(tf.matmul(X, w), b, name="z")   
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0)) #表示实在定义relu这个Variable
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close()
~~~

推荐方法4: 使用getVariable（版本2:在使用Variable的函数内部定义）

~~~python
reset_graph()

def relu(X):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1) 
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  
    b = tf.Variable(0.0, name="bias")  
    z = tf.add(tf.matmul(X, w), b, name="z")  
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu9", tf.get_default_graph())
file_writer.close()
~~~

方法5: 与方法4思路相同，代码略繁琐一点

~~~python
reset_graph()

def relu(X):
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("", default_name="") as scope:
    first_relu = relu(X)     # create the shared variable
    scope.reuse_variables()  # then reuse it
    relus = [first_relu] + [relu(X) for i in range(4)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
file_writer.close()
~~~

