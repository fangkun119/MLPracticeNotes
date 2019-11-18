代码RNN.09 创造性RNN （预测序列下一个值 & 序列生成）

~~~python
n_steps = 20

with tf.Session() as sess:  
   # 载入06或07代码训练出来的模型
    saver.restore(sess, "./my_time_series_model") 
    # 用0值构造一个种子序列
    sequence = [0.] * n_steps
   # 迭代3000次，生成一个长度为3000的序列
    for iteration in range(300):
	# 取后序列末尾n_steps=20个序列值、作为一个样本
        X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
	# 交给模型去预测接下来的序列值(例如用seq[n-19,n], 预测seq[n-18, n+1]
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
	# 将预测值的最后一个值追加到序列中
        sequence.append(y_pred[0, -1, 0])

#可视化预测结果
plt.figure(figsize=(8,4))
plt.plot(np.arange(len(sequence)), sequence, "b-")
plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
~~~

￼

~~~python
# 用0值构造种子序列，和用真实值构造的种子序列，生成的预测序列还是不同的
with tf.Session() as sess:
    saver.restore(sess, "./my_time_series_model")

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
save_fig("creative_sequence_plot")
plt.show()
~~~

￼






