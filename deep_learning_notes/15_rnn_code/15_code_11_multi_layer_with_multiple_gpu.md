代码RNN.11 将多层RNN中组装在MultiRNNCell中的BasicRNNCell分配到不同的GPU

错误方法（不会生效，因为BasicRNNCell是工厂）

~~~python
# DO NOT DO THIS
with tf.device("/gpu:0"):   # BAD! This is ignored
    layer1 = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
with tf.device("/gpu:1"):   # BAD! Ignored again
    layer2 = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
~~~

正确方法

~~~python
import tensorflow as tf

class DeviceCellWrapper(tf.nn.rnn_cell.RNNCell):
  def __init__(self, device, cell):
    self._cell = cell
    self._device = device

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    with tf.device(self._device):
        return self._cell(inputs, state, scope)

reset_graph()

n_inputs = 5
n_steps = 20
n_neurons = 100

X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs]) 

devices = ["/cpu:0", "/cpu:0", "/cpu:0"]   # replace with ["/gpu:0", "/gpu:1", "/gpu:2"] if you have 3 GPUs
cells = [DeviceCellWrapper(dev, tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)) for dev in devices]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(sess.run(outputs, feed_dict={X: np.random.rand(2, n_steps, n_inputs)}))
~~~
