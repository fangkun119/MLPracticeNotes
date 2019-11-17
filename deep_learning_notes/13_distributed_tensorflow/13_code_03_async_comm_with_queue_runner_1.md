代码_分布式TF.03.用TensorFlow队列进行异步通信.新方法_用QueueRunner和Coordinators

~~~python
reset_graph()

# 定义FileName队列(FIFOQueue)相关操作
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
						min_after_dequeue=2, #用来保证有一定随机性的参数
						dtypes=[tf.float32, tf.int32], shapes=[[2],[]],
						name="instance_q", 
						#TF根据这个名字以及tf.container(如果有)名字在cluster的各个client中共享队列
						shared_name="shared_instance_q") 

# 定义数据入队的操作
enqueue_instance = instance_queue.enqueue([features, target])
close_instance_queue = instance_queue.close()

# 定义数据出队的操作
minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

# 定义QueueRunner，Coordinator
n_threads = 5
queue_runner = tf.train.QueueRunner(instance_queue,  [enqueue_instance] * n_threads)
coord = tf.train.Coordinator()

with tf.Session() as sess:
    # 创建filename队列(FIFOQueue)并加载和入队fileName
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    # 关闭filename队列(FIFOQueue): 向其他会话发出信号，这样不会再有数据入队
    sess.run(close_filename_queue)
    # 启动QueueRunner和Coordinator，多线程加载数据入队
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    try:
        while True:
            # 数据出队
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")
~~~