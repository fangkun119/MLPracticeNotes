#分布式TensorFlow

## 1.在多GPU卡的机器上安装TensorFlow
查询GPU卡的兼容性：[https:developer.nvidia.com/cuda-gpus](https:developer.nvidia.com/cuda-gpus)<br/>
GPU托管服务：AWS<br/>
Google Cloud ML：[https:///cloud.google.com/ml](https:///cloud.google.com/ml)<br/>
自己买GPU卡：[https://goo.gl/gCtSAn](https://goo.gl/gCtSAn)<br/>
检查CUDA是否安装成功，以及可用的GPU卡：nvdia-smi
安装步骤：P280

## 2.管理整个TF Session的GPU RAM占用
默认：TF会占用GPU所有RAM（无法同时启动两个TF）<br/>
指定TF启动时使用哪个GPU：P283 <br/>
指定TF启动时只占用一定比例的CPU内存：P283 <br/>
指定TF按需使用GPU内存：P284 <br/>

## 3.自动分配代码块在哪块GPU上运行（动态配置方法）
还没有开源，只在Google内部使用，另外似乎总是可以通过手动配置来更优化地使用硬件

## 4.用with tf.device指定节点在哪块GPU上运行（简单配置方法）

**(1) TF分配规则**

priority 1: 节点在某个设备上运行过，则继续在这个设备上运行<br/>
priority 2: 节点包裹在`with tf.device(..):中，根据tf.device(…)`决定运行在哪个设备上<br/>
priority 3: 使用默认值“/cpu:0"<br/>

**(2) 使用`with tf.device(${device_id})`配置节点运行的硬件**

~~~python
with tf.device("/cpu:0"):
	a = tf.Variable(1.0, name="a")
	b = a + 2
c = a + b
~~~

**(3) 使用`with tf.device(${device_decide_func})`来根据某种特性来决定节点在哪个硬件上执行**

~~~python
def viriables_on_cpu(op): 
	if op.type == "Variable":
		return "/cpu:0"
	else:
		return "/gpu:0"
with tf.device(viriables_on_cpu):
	a = tf.Variable(3.0)
	b = tf.constant(4.0)
	c = a + b
~~~

**(4) 在日志中增加`log_device`的打印，查看节点时在哪个设备上运行的**

~~~python
config = tf.ConfigProto()
config.log_device_placement = True
sess = tf.Session(config = config) 
c.initializer.run(session=sess)
~~~

**5.注意事项：并不是所有操作(OP)都能运行在所有设备上**<br/>
只有TF为这个<OP, device>实现了具体的操作内核时，该OP才能在这个device上运行, 例如整数变量没有GPU内核：

* 要么运行在CPU上，
* 要么改成浮点数，
* 要么开启allow_soft_placement配置，容许TF把这个操作从GPU改到CPU上

**例子**：会抛InvalidArgumentError的代码

~~~python
with tf.device("/gpu:0"):
	i = tf.Variable(3) 
sess = tf.Session()
sess.run(i.initializer)
例子：开启allow_soft_placement配置，将操作从GPU改到CPU上
with tf.device("/gpu:0"):
	i = tf.Variable(3) 
config = tf.ConfigProto()
config.allow_soft_placement = True
sess.run(i.initiazlier)
~~~

## 6.并行操作：

TF根据操作(OP)间的依赖关系，让能并行的OP并行执行<br/>

**线程池**：不同操作使用inter-op线程池来并发执行；同一个操作的操作内核中的多线程代码使用intra-op线程池来并发执行<br/>
**配置：**<br/>

* `inter_op_parallelism_threads`：`inter-op`线程池线程数
* `intra_op_parallelism_threads`：`intra-op`线程池线程数
* `use_per_session_threads`:  后续session各自使用自己的线程池、还是复用第一个session创建的`inter_op`线程池

**手动添加依赖：**TF根据数据读取来判断依赖关系，有时候想人工增加新的依赖<br/>

**例如**：想延后执行某个很占内存的操作C，等另外两个大内存操作A、B结束后再执行，虽然C并不需要A、B的计算结果<br/>

~~~python
a = tf.constant(1.0)
b = a + 2.0
with tf.control_depnedencies([a, b]):
	c = tf.constant(3.0)
~~~

## 7.单设备（简单本地会话）

变量状态由所属会话管理，不会共享（即使这些会话运行的是同一个数据流图，每个会话也是各自有自己的变量副本）

**例子:**

~~~python
import tensorflow
c = tf.constant("Hello distributed TensorFlow!")
server = tf.train.Server.create_local_server()
with tf.Session(server.target) as sess:
    print(sess.run(c))
~~~

## 8.TensorFlow Cluster：多设备/服务器TensorFlow

**步骤：**<br/>

1. 定义集群（1个集群1-N台机器）<br/>
2. 定义作业（1个作业一组作用相同的任务；其中作业"ps"表示追踪模型参数；作为“worker”通常表示执行计算）<br/>
3. 	绑定变量（包含模型参数）到作业

	> 可以手动分配
	> 可以用tf.train.replica_device_setter来自轮流分配
4. 绑定操作（OP）到作业，需要手动绑定，如果没有手动绑定会绑定到默认值上

**例子：**<br/>
[代码_分布式TF.01.定义集群及作业，并手动/自动绑定模型参数，手动绑定OP到集群-作业-机器-device上](13_distributed_tensorflow/13_code_01_distributed_job.md)

## 9.对于TensorFlow Cluster，变量(Variable)在多设备/服务器之间是共享的

* 变量状态不再由会话自己管理，而是由集群的资源容器管理
* 当一个客户端会话创建一个名为x的变量时，它会对同一集群的其他会话可用（即使两个会话连接至不同的服务器）

例子：两个不同的客户端，连接到同一个Cluster的机器中时，Variable在各个客户端之间时共享的

~~~python
#server_addr是一个类似grpc://machine-a.example.com:2222的字符串，假定传入的server_addr已经被配置到了同一个Cluster中(参考附录12.01.定义集群及作业)
x = tf.Variable(0.0, name="x")
increament_x = tf.assign(x, x+1)
server_addr = sys.argv[1]  
with tf.Session (server_addr) as sess:
	if sys.argv[2:] == ["init"]:
		sess.run(x.initializer)
	sess.run(increment_x)
	print(x.eval())

#启动两个客户端，会发现变量x在各个客户端之间是共享的
#$ python3 simple_client.py grpc://machine-a.example.com:2222 init
#$ python3 simple_client.py grpc://machine-b.example.com:2222 
~~~

## 10.如果想在TensorFlow Cluster中阻止变量共用（创建一个独立的计算）

有两个方法：

1. 使用tf.variable_scope，例子如下

	~~~python
	with tf.variable_scope("my_problem_1"):
		sess.run(increment_x)
	~~~

2. 使用tf.container，例子如下

	~~~python
	with tf.container("my_problem_1"):
		sess.run(increment_x) 
	~~~

其中更推荐使用tf.conftainer，有两个原因

1. 能为容器起一个名字，而不是使用默认的“”
2. 能够方便地重置容器名，同时将容器内之前用过的资源全部释放

	~~~python
	tf.Session.reset("grpc://machine-a.example.com:2222", ["my_problem_1"])
	~~~

## 11.用tf.decode_csv解析Reader从CSV读取的数据(OLD Approach)

~~~python
default1 	= tf.constant([5.]) # 类型为浮点数、缺省默认值为5
default2 	= tf.constant([6])	# 类型为整数、缺省默认值为6
default3 	= tf.constant([7])	# 类型为整数、缺省默认值为7
# key, value = reader.read(filename_queue)   # 获得csv的方法，key为${file_name}:${line_num}，val为csv的行内容
dec 		= tf.decode_csv(tf.constant("1.,,44"), record_defaults=[default1, default2, default3])
with tf.Session() as sess:
	print(tf.constant("1.,,44"))
    	print(sess.run(dec))
# Tensor("Const_4:0", shape=(), dtype=string)
# [1.0, 6, 44] 
# 其中1.0，44来自CSV的行，6来自默认值
~~~

## 12.用异步通信加快模型训练PIPE Line的运行速度（含数据加载例子）

使用场景举例：不希望准备好所有Mini-Batch后再开始训练，希望一边用前一个Mini-Batch训练，一边准备后面一个<br/>

**方案<br/>**

step1 	定义一个TensorFlow Cluster<br/>
step2	设置一个队列`tf.FIFOQueue/tf.RandomShufferQueue/tf.PaddingFifoQueue`，并且起名例如`shared_name = "shared_q")`<br/>
step3 	启动一个Client连接到Cluster，负责生成Mini-Batch到队列中<br/>
step4 	启动一个Client连接到Cluster，负责消费队列中的Mini-Batch用于训练模型<br/>

**代码<br/>**

[代码\_分布式TF.02.用TensorFlow队列进行异步通信.使用Reader将数据插入队列（旧方法）](13_distributed_tensorflow/13_code_02_async_comm_old_way.md)<br/>
[代码\_分布式TF.03.用TensorFlow队列进行异步通信.新方法_用QueueRunner和Coordinators](13_distributed_tensorflow/13_code_03_async_comm_with_queue_runner_1.md)<br/>
[代码\_分布式TF.04.用TensorFlow队列进行异步通信.新方法_用QueueRunner和Coordinators （函数封装）](13_distributed_tensorflow/13_code_04_async_comm_with_queue_runner_2.md)<br/>
[代码\_分布式TF.05.设置队列操作的TimeOut](13_distributed_tensorflow/13_code_05_timeout.md)<br/>
[代码\_分布式TF.06 用TF1.4的DATA API来加载数据](13_distributed_tensorflow/13_code_06_load_data_with_tf1_4_api.md)<br/>

**补充要点<br/>**

* TF根据`Queue`的`shared_name`，以及`tf.container`的`name`（如果使用了`container`)来在(同一个`cluster`的)不同`session`间共享队列
* `PaddingFifoQueue`会根据形状用0值来填充缺失的字段和`record`等（取决于张量的维度）；
* `RandomShufferQueue`可以在一定程度上做到数据随机出队
* 关闭队列可以向其他会话发出信号，这样就不再会有数据入队
* 数据出队除了可以使用`dequeue`，还可以使用`dequeue_many`，注意队列中`record`不够时`dequeue_many`会阻塞
* 可以通过`config.operation_timeout_in_ms = 1000`外加`with tf.Session(config=config) as sess`的方式来设置队列操作的`timeout`

##13.加载数据(DATA API)(TenserFlow 1.4)：生成数据的API

**1.数据流图定义**

~~~python
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10)) #生成1,2,3,…,9
dataset = dataset.repeat(3).batch(7) #将1,2,3,..9重复3遍、再以7个一组分成4个batch
iterator = dataset.make_one_shot_iterator() #生成迭代器
next_element = iterator.get_next() #定义获取迭代器下一个元素的操作
~~~

**2. 从迭代器读取数据 (在with tf.Session() as sess:中运行，要捕捉**

~~~python
print(next_element.eval()
~~~

**3. 在同一个sess.run中，iterator只会往前走一步（不论next_element被调用多少次）**

~~~python
with tf.Session() as sess:
	#两个next_element取出的内容相同，要到下次sess.run才会取下一个
	…
   print(sess.run([next_element, next_element])) 
~~~

**4. `interleave()`：用来从几个batch中交替采样**

**5. 用Tensorflow 1.4的DATA API来加载数据**

~~~python
dataset = dataset.skip(1).map(decode_csv_line)
it = dataset.make_one_shot_iterator()
X, y = it.get_next()
~~~

以上5个小节的完整代码：[代码_分布式TF.06 用TF1.4的DATA API来加载数据](13_distributed_tensorflow/13_code_06_load_data_with_tf1_4_api.md)


## 14.其他API

~~~python
string_input_producer
input_producer
range_input_producer
slice_input_producer
shuffle_batch
batch
batch_join
shuffle_batch_join
~~~

P304

## 15.数据并行方案

**1. 一台设备训练一个神经网络：例如在50个有2个GPU的服务器上训练100个神经网络<br/>**

用途1: 超参数调参（同时训练多个神经网络）<br/>
用途2: 处理大QPS模型预测（也可以使用TensorFlow Serving，https://tensorflow.github.io/serving/）<br/>

**2. 图内复制**<br/>

用途：针对大型神经网络集合体进行并行巡礼那，在不同的设备上部署神经网络，汇总每个神经网络的预测，从而得到集合体的预测<br/>
方法：创建一个大图，包含每一个神经网络，分别配置在不同的设备上，最后再再某个设备上创建会话让器负责汇总操作<br/>

**3. 图间复制**<br/>

用途：针对大型神经网络集合体进行并行训练时，在不同的设备上部署神经网络，汇总每个神经网络的预测，从而得到集合体的预测<br/>

方法：为每个神经网络创建一个独立的图，并自己处理图之间的拷贝（例如使用队列）

> 底部客户端：读取输入数据，存入输入队列<br/>
> 顶部客户端：读取预测队列的数据，汇总成整个集合体的预测<br/>
> 中间：负责处理神经网络预测，从输入队列渠道数据，进行预测，放入预测队列<br/>

备注：其中汇总集合体预测时需要以后超时机制，参考[代码_分布式TF.05.设置队列操作的TimeOut
](13_distributed_tensorflow/13_code_05_timeout.md)

## 16.模型并行方案

用途：希望在多个设备上运行一个单独的神经网络，即将模型分为不同的块并在不同设备上运行这些块，被称为模型并行化<br/>

要点：<br/>

* 全联接网络没有必要切割，效果不好
￼* 需要依据神经网络的架构来切割，效果才够
￼* 对RNN进行切割，因为每一层都很复杂，网络开销是值得的

## 17.数据并行化

1. 思路：每台设备上拷贝一份神经网络，在多个副本上同时训练（使用不同的mini-batch），然后汇总梯度来更新模型参数
￼
2. 更新方法

	* 同步更新：等待所有副本都得到梯度后在进行参数更新，缺点是某些设备可能慢（可以考虑忽略最慢的几个副本）
	* 异步更新：每个副本完成计算立即更新模型参数（不汇总、不平均、不等待和同步），缺点是存在梯度过期的问题（减轻梯度过期影响的方法包括：(1)降低学习率(2)丢弃或降低过期梯度(3)调整小批量大小(4)最开始的几个全数据集只用一个副本（热身阶段）
	* 目前研究进展（2016 Goolge Paper）：同步更新最有效，然后异步更新也是热门的研究领域

3. 带宽饱和问题：虽然没有全联接层，带宽仍然会被饱和<br/>

	原因：数据并行化需要在每个训练步骤开始阶段，把模型参数传给每一个副本，步骤结束时再传回梯度。这意味着添加额外GPU可能也不会提高性能<br/>

	经验：

	* 模型小但是训练集大时：最好在单GPU单机器上训练；大密度模型情况更严重；
	* 小模型(使用数据并行化带来的收益少)和大型稀疏模型，因为梯度基本为零，可以有效通信
	* 2016年的硬件和技术情况：密集模型超过几十个GPU、稀疏模型超过几百个GPU时，饱和现象出现性能显著下降。以下模型都是稀疏模型，性能不错
	    * 神经机器翻译：8个GPU、提速6倍
	    * Inception/ImageNet：50个GPU，提速32倍
	    * RankBrain：500个GPU，提速300倍

4. 缓解方法：

	* GPU分组在少量几个服务器上，而不是分散在很多服务器上：减少网络开销
	* 在多个参数服务器间分片参数：如前面所述
	* 将模型参数浮点精度从tf.float32降低到tf.float16，数据减半、但不会影响收敛速度和性能（16位是最低最低训练精度，但是训练完成后可以进一步降低到8位，量化神经网络，对于手机部署和运行于预训练模型非常有用）

## 19. 并行化方案 

首先确定是图内拷贝还是图间拷贝，再确定是同步更新还是异步更新

* 图内拷贝 + 同步更新：创建一个完整的大图（包含所有模型副本），同时有几个节点在汇总所有梯度，并将其赋给一个优化器
* 图内拷贝 + 异步更新：创建一个大图，但是每个副本都有一个优化器（由副本自带的线程运行）
* 图间拷贝 + 异步更新：运行多个独立客户端（在不同进程），每个模型看上去是独立训练的，其实参数是共享的
* 图间拷贝 + 同步更新：运行多个客户端，每个客户端都会根据共享参数对模型副本进行训练（代码：用SyncReplicasOptimizer封装优化器例如MomentiumOptimizer）、每个副本看起来与非并行化时代码很相似，其实在后台优化器会将梯度传给一组队列（每个Variable一个），chief副本的一个SyncReplicasOptimizer会读取这个队列，汇总并使用读到的梯度，然后给每个副本的令牌队列写入一个令牌，表示这些副本可以继续运行下一个梯度

代码：见作者git
