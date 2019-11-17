代码_分布式TF.01.定义集群及作业，并手动/自动绑定模型参数，手动绑定OP到集群-作业-机器-device上

~~~python
# 定义集群（有相关的作业组成）
cluster_spec = tf.train.ClusterSpec({
    "ps": [
        "127.0.0.1:2221",	# /job:ps/task:0
        "127.0.0.1:2222",  # /job:ps/task:1
    ],
    "worker": [
        "127.0.0.1:2223",  # /job:worker/task:0
        "127.0.0.1:2224",  # /job:worker/task:1
        "127.0.0.1:2225",  # /job:worker/task:2
    ]})
# 启动集群中的Server（作业）
task_ps0 = tf.train.Server(cluster_spec, job_name="ps", task_index=0)
task_ps1 = tf.train.Server(cluster_spec, job_name="ps", task_index=1)
task_worker0 = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
task_worker1 = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
task_worker2 = tf.train.Server(cluster_spec, job_name="worker", task_index=2)
reset_graph()
# 绑定操作节点到集群的作业，作业机器
# 手动绑定模型参数(变量)节点到作业ps
with tf.device("/job:ps"):
    a = tf.Variable(1.0, name="a")
# 绑定OP节点到作业worker
with tf.device("/job:worker"):
    b = a + 2
with tf.device("/job:worker/task:1"):
    c = a + b
# 启动客户端
with tf.Session("grpc://127.0.0.1:2221") as sess:
    sess.run(a.initializer)
    print(c.eval())

# 下面开始是例子2，演示使用参数自动分片的方法来为模型参数绑定节点（因为模型参数量比较大，通常需要自动绑定）
# 先重置数据流图
reset_graph()
# 参数分片：ps作业用来跟踪模型参数，worker作用用来执行OP
# 对于模型参数，告诉TF有几个ps作业，TF会自动轮流分配
# 对于OP，需要手动来指定分配到那个作业（或作业/机器，或作业/机器/设备）上，如果没指定，就使用缺省值
with tf.device(tf.train.replica_device_setter(ps_tasks=2, ps_device="/job:ps", worker_device="/job:worker")):
    v1  = tf.Variable(1.0, name="v1")  	# pinned to /job:ps/task:0	(defaults to /cpu:0)
    v2 = tf.Variable(2.0, name="v2") 	# pinned to /job:ps/task:1	(defaults to /cpu:0)
    v3 = tf.Variable(3.0, name="v3") 	# pinned to /job:ps/task:0	(defaults to /cpu:0)
    s = v1 + v2            	# pinned to /job:worker (defaults to task:0/cpu:0)
    with tf.device("/task:1"):
        p1 = 2 * s         	# pinned to /job:worker/task:1 (defaults to /cpu:0)
        with tf.device("/cpu:0"):
            p2 = 3 * s 	# pinned to /job:worker/task:1/cpu:0
# 开启log_device日志打印
config = tf.ConfigProto()
config.log_device_placement = True
# 启动客户端
with tf.Session("grpc://127.0.0.1:2221", config=config) as sess:
    v1.initializer.run()
~~~


