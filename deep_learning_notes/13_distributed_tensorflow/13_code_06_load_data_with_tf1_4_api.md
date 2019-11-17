代码_分布式TF.06 用TF1.4的DATA API来加载数据

1. 使用iterator

~~~python
tf.reset_default_graph()
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10)) #生成1,2,3,…,9
dataset = dataset.repeat(3).batch(7) #将1,2,3,..9重复3遍、再以7个一组分成4个batch
iterator = dataset.make_one_shot_iterator() #生成迭代器
next_element = iterator.get_next() #定义获取迭代器下一个元素的操作
with tf.Session() as sess:
    try:
        while True:
            print(next_element.eval()) #运行数据流图，取数据
    except tf.errors.OutOfRangeError:
        print("Done")
~~~

输出
# [0 1 2 3 4 5 6]
# …
# [1 2 3 4 5 6 7]
# [8 9]
# Done

~~~python
#在一个sess.run中，iterator只会往前走一步（不论next_element被调用多少次）
with tf.Session() as sess:
    try:
        while True:
            print(sess.run([next_element, next_element]))
    except tf.errors.OutOfRangeError:
        print("Done")
~~~

# 输出 # [array([0, 1, 2, 3, 4, 5, 6]), array([0, 1, 2, 3, 4, 5, 6])]
# …
# [array([1, 2, 3, 4, 5, 6, 7]), array([1, 2, 3, 4, 5, 6, 7])]
# [array([8, 9]), array([8, 9])]
# Done


2. 用新的API(iterator)加载数据

~~~python
tf.reset_default_graph()
filenames = ["my_test.csv"]
dataset = tf.data.TextLineDataset(filenames)

def decode_csv_line(line):
    x1, x2, y = tf.decode_csv(
        line, record_defaults=[[-1.], [-1.], [-1.]])
    X = tf.stack([x1, x2])
    return X, y

dataset = dataset.skip(1).map(decode_csv_line)
it = dataset.make_one_shot_iterator()
X, y = it.get_next()

with tf.Session() as sess:
    try:
        while True:
            X_val, y_val = sess.run([X, y])
            print(X_val, y_val)
    except tf.errors.OutOfRangeError as ex:
        print("Done")
~~~

3. Interleave
The interleave() method is powerful but a bit tricky to grasp at first. The easiest way to understand it is to look at an example:

~~~python
tf.reset_default_graph()
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.interleave(
    	lambda v: tf.data.Dataset.from_tensor_slices(v),
    	cycle_length=3,
    	block_length=2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(next_element.eval(), end=",")
    except tf.errors.OutOfRangeError:
        print("Done")

# 输出:
# 0,1,7,8,4,5,2,3,9,0,6,7,4,5,1,2,8,9,6,3,0,1,2,8,9,3,4,5,6,7,Done
~~~

说明:
Because cycle_length=3, the new dataset starts by pulling 3 elements from the previous dataset: that's [0,1,2,3,4,5,6], [7,8,9,0,1,2,3] and [4,5,6,7,8,9,0]. Then it calls the lambda function we gave it to create one dataset for each of the elements. Since we use Dataset.from_tensor_slices(), each dataset is going to return its elements one by one. Next, it pulls two items (since block_length=2) from each of these three datasets, and it iterates until all three datasets are out of items: 0,1 (from 1st), 7,8 (from 2nd), 4,5 (from 3rd), 2,3 (from 1st), 9,0 (from 2nd), and so on until 8,9 (from 3rd), 6 (from 1st), 3 (from 2nd), 0 (from 3rd). Next it tries to pull the next 3 elements from the original dataset, but there are just two left: [1,2,3,4,5,6,7] and [8,9]. Again, it creates datasets from these elements, and it pulls two items from each until both datasets are out of items: 1,2 (from 1st), 8,9 (from 2nd), 3,4 (from 1st), 5,6 (from 1st), 7 (from 1st). Notice that there's no interleaving at the end since the arrays do not have the same length.


 
