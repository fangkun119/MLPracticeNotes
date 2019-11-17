代码_分布式TF.04.用TensorFlow队列进行异步通信.新方法_用QueueRunner和Coordinators （函数封装）

~~~python
reset_graph()

# 数据加载函数：从filename_queue取出一个file_name，加载该文件，将数据入队到instance_queue中
def read_and_push_instance(filename_queue, instance_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
    features = tf.stack([x1, x2])
    enqueue_instance = instance_queue.enqueue([features, target])
    return enqueue_instance

# 定义FileName队列(FIFOQueue)相关操作
filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

# 定义数据队列(RandomShuffleQueue，record出队会有一定程度的随机性)
instance_queue = tf.RandomShuffleQueue(
							capacity=10, 
							min_after_dequeue=2, #用来保证有一定随机性的参数
							dtypes=[tf.float32, tf.int32], 
							shapes=[[2],[]],
							name="instance_q", 
							#TF根据这个名字以及tf.container(如果有)名字在cluster的各个client中共享队列
							shared_name="shared_instance_q") 
# 定义数据出队的操作
minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

# 定义数据入队的操作，由QueueRunner来调用
read_and_enqueue_ops = [read_and_push_instance(filename_queue, instance_queue) for i in range(5)]
queue_runner = tf.train.QueueRunner(instance_queue, read_and_enqueue_ops)

with tf.Session() as sess:
    # 文件名入队
    sess.run(enqueue_filename, feed_dict={filename: "my_test.csv"})
    sess.run(close_filename_queue)
    # 数据入队（用N个线程）
    coord = tf.train.Coordinator()
    enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)
    # 数据出队、用于模型训练（由主线程来执行）
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")
~~~