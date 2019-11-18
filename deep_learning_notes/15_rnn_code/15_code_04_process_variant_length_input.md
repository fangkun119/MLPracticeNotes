代码RNN.04 RNN代码技巧：处理变长输入

~~~python
n_steps = 2  # 时间片数
n_inputs = 3  # 样本特征数 
n_neurons = 5  # 神经元数量

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)

seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
 init = tf.global_variables_initializer()

X_batch = np.array([
        # step 0    step 1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

# 输出每个样本在各个时间片的输出
print(outputs_val)
# [[[-0.9123188   0.16516446  0.5548655  -0.39159346  0.20846416]
#   [-1.          0.956726    0.99831694  0.99970174  0.96518576]]
#  [[-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]
#   [ 0.          0.          0.          0.          0.        ]]
#  [[-0.99999976  0.8967997   0.9986295   0.9647514   0.93662   ]
#   [-0.9999526   0.9681953   0.96002865  0.98706263  0.85459226]]
#  [[-0.96435434  0.99501586 -0.36150697  0.9983378   0.999497  ]
#   [-0.9613586   0.9568762   0.7132288   0.97729224 -0.0958299 ]]]

# 输出每个样本最后一次输出
print(states_val)
# [[-1.          0.956726    0.99831694  0.99970174  0.96518576]
#  [-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]
#  [-0.9999526   0.9681953   0.96002865  0.98706263  0.85459226]
#  [-0.9613586   0.9568762   0.7132288   0.97729224 -0.0958299 ]]
~~~