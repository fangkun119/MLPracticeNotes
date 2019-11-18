代码RNN.14 RNN 自然语言处理.Embedding

~~~
This section is based on TensorFlow's Word2Vec tutorial.

from six.moves import urllib

import errno
import os
import zipfile

WORDS_PATH = "datasets/words"
WORDS_URL = 'http://mattmahoney.net/dc/text8.zip'

def mkdir_p(path):
    """用来支持python 2, python > 3.2时用os.makedirs(path, exist_ok=True)就可以"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# 1. 取样本集，并切词，得到单词明文列表(words)
def fetch_words_data(words_url=WORDS_URL, words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, "words.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(words_url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()

words = fetch_words_data()
words[:5]
# ['anarchism', 'originated', 'as', 'a', 'term']


# 2. 构建单词的序号表示(data)，单词序号到单词明文的映射(vocabulary)
from collections import Counter

vocabulary_size = 50000
vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])

" ".join(words[:9]), data[:9]
('anarchism originated as a term of abuse first used',  array([5234, 3081,   12,    6,  195,    2, 3134,   46,   59]))
" ".join([vocabulary[word_index] for word_index in [5241, 3081, 12, 6, 195, 2, 3134, 46, 59]])
'cycles originated as a term of abuse first used'
words[24], data[24]
('culottes', 0)


# 3. 生成batch
from collections import deque

def generate_batch(batch_size, num_skips, skip_window):		# batch_size = 8, num_skips = 2, skip_window = 1
    global data_index
   assert batch_size % num_skips == 0
   assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=[batch_size], dtype=np.int32) 	# 8		batch: batch_size 长度的数组
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)	# 8*1:	labels: batch_size 行 1 列 的向量
    span = 2 * skip_window + 1 							# 3: 		span = 2*skip_window + 1  # [ skip_window target skip_window ] 
    buffer = deque(maxlen=span)							# buffer：双向队列，长队是span
    for _ in range(span):									# 填充span = 2*skip_window + 1 = 3个词到buffer中
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data) 
    for i in range(batch_size // num_skips):					# i = [0, 4)   /*4=8/2; 8=batch_size; 2=num_skips*/
        target = skip_window 								# 	target = skip_window = 1  (buffer正中的位置)
        targets_to_avoid = [ skip_window ]					#	target 所在位置先加入 targets_to_avoid
        for j in range(num_skips):							#	j = [0, 2)   /*2=num_skips*/
            while target in targets_to_avoid:					#		target随机，但不能在avoid范围内（即不在buffer正中，也没被选过）
                target = np.random.randint(0, span)				#		
            targets_to_avoid.append(target)					#		target选好之后，也要加到target_to_avoid中，避免以后被选中
            batch[i * num_skips + j] = buffer[skip_window]			#		当前位置 batch[cur_idx = i * 2 + j] 的值，设为buffer正中位置的值
            labels[i * num_skips + j, 0] = buffer[target]			#		当前位置 labels[cur_idx = i * 2 + j, 0] 的值，设为buffer[rand_target]的值
        buffer.append(data[data_index])						#	填充一个词到buffer中
        data_index = (data_index + 1) % len(data)
    return batch, labels


# 生成batch的例子
np.random.seed(42)
data_index = 0
batch, labels = generate_batch(8, 2, 1)

" ".join(words[:9]), data[:9]
('anarchism originated as a term of abuse first used',  array([5234, 3081,   12,    6,  195,    2, 3134,   46,   59]))

" ".join([vocabulary[word_index] for word_index in [5241, 3081, 12, 6, 195, 2, 3134, 46, 59]])
'cycles originated as a term of abuse first used'

batch, [vocabulary[word] for word in batch]
# (array([3081, 3081,   12,   12,    6,    6,  195,  195], dtype=int32),
# ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term'])

labels, [vocabulary[word] for word in labels[:, 0]]
(array([[  12],
        [5234],
        [   6],
        [3081],
        [  12],
        [ 195],
        [   2],
        [   6]], dtype=int32),
 ['as', 'anarchism', 'a', 'originated', 'as', 'term', 'of', 'a'])

# batch，label对应关系 (batch_size=8;  num_skips=2;  skip_window=1)
# 'originated' -> 'as'    			/*as 		是 orginated 后面的词*/
# 'originated' -> 'anarchism'		/*anarchism 	是 orginated 前面的词*/
# 'as' -> 'a',					/*a 			是 as 后面的词*/
# 'as' -> 'originated',			/*originated 	是 as 前面的词*/
# 'a' -> 'as', 					/*as			是 a 前面的词*/
# 'a' -> 'term',				/*term		是 a 后面的词*/
# 'term' -> 'of' 				/*of			是 term 后面的词*/
# 'term' -> 'a'					/*a			是 term 前面的词*/

# 4. 模型数据流图
# batch 参数（参考上面的例子）
batch_size 		= 128	# 一个batch有128个'词'->'label'，label是词两边的词
embedding_size 	= 128  	# 生成的embedding向量的长度
skip_window = 1       		# 考虑每个词左右两边多少个词
num_skips = 2         		# 每个词生成几个label

# 验证参数
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     			# 16个随机的词用于验证：Random set of words to evaluate similarity on.
valid_window = 100  		# 选numeric ID最低的词（词频最高）：Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  #从100中随机选16个numeric num id用于验证
num_sampled = 64   		# 选64个负例 Number of negative examples to sample.

# 学习率
learning_rate = 0.01
reset_graph()

# 输入数据
train_labels 	= tf.placeholder(tf.int32, shape=[batch_size, 1])		# batch label （见3.*的例子）
valid_dataset 	= tf.constant(valid_examples, dtype=tf.int32)		# 16个用于validation的词的numeric ID

# 50000个词 -> embedding成一个150维向量，按正态分布随机初始化
vocabulary_size = 50000
embedding_size = 150

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)  #[50000, 150]矩阵
embeddings = tf.Variable(init_embeds)

train_inputs = tf.placeholder(tf.int32, shape=[None])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)   #train_input为词的numeric_id，返回结果是150维的embedding向量

# 初始化权重矩阵(为正太分布随机数)，NCE偏差的向量(为0)
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],  stddev=1.0 / np.sqrt(embedding_size))) #权重矩阵[50000*1]矩阵
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))    #50000维向量

# NCE损失值计算: tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, vocabulary_size))

# 优化器，最小化损失值：Construct the Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# Mini-Batch与所有Embedding的余弦相似度：Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)) #所有embeddings向量的平方根均值
normalized_embeddings = embeddings / norm   #正则化
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) #查找mini-batch中的embedding
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) #余弦相似度

# 初始化全局变量
init = tf.global_variables_initializer()

# 5. 模型训练
num_steps = 10001

with tf.Session() as session:
    init.run()

    average_loss = 0
    for step in range(num_steps):
       	print("\rIteration: {}".format(step), end="\t")
       	batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
       	feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

      	# We perform one update step by evaluating the training op (including it in the list of returned values for session.run()
        	_, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
        	average_loss += loss_val

        	if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocabulary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = vocabulary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()
    np.save("./my_final_embeddings.npy", final_embeddings)

# 6. embedding可视化
def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [vocabulary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
~~~

￼



