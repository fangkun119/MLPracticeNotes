代码_分布式TF.02.用TensorFlow队列进行异步通信.使用Reader将数据插入队列（旧方法）

~~~python
reset_graph()

test_csv = open("my_test.csv", "w")
test_csv.write("x1, x2 , target\n")
test_csv.write("1.,, 0\n")
test_csv.write("4., 5. , 1\n")
test_csv.write("7., 8. , 0\n")
test_csv.close()

# 定义FileName队列(FIFOQueue)
filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

# 定义csv加载操作
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
features = tf.stack([x1, x2])

# 定义数据队列(RandomShuffleQueue，record出队会有一定程度的随机性)
instance_queue = tf.RandomShuffleQueue(
capacity=10, 
# 用来保证有一定随机性的参数
min_after_dequeue=2, 
dtypes=[tf.float32, tf.int32], 
shapes=[[2],[]],
name="instance_q", 
#TF根据这个名字以及tf.container(如果有)名字在cluster的各个client中共享队列
shared_name="shared_instance_q" 
)

# 定义入队操作节点
enqueue_instance = instance_queue.enqueue([features, target])
# 定义队列关闭节点
close_instance_queue = instance_queue.close()
# 定义出队节点
minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

with tf.Session() as sess:
# 创建filename队列(FIFOQueue)并加载和入队fileName
sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
# 关闭filename队列(FIFOQueue): 向其他会话发出信号，这样不会再有数据入队
sess.run(close_filename_queue)
# 数据入队直到全部入完
try:
while True:
sess.run(enqueue_instance)
except tf.errors.OutOfRangeError as ex:
print("No more files to read")
# 关闭队列(RandomShuffleQueue): 向其他会话发出信号，这样不会再有数据入队
sess.run(close_instance_queue)
# 数据出队(RandomShuffleQueue)直到全部出队
try:
while True:
print(sess.run([minibatch_instances, minibatch_targets]))
except tf.errors.OutOfRangeError as ex:
print("No more training instances")
~~~
