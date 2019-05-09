#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import xrange  
import tensorflow as tf
import jieba
from itertools import chain

"""第一步：读取数据，用jieba进行分词，去除停用词，生成词语列表。"""
def read_data(filename):
    f = open(filename, 'r',  encoding='utf-8')
    stop_list = [i.strip() for i in open('ChineseStopWords.txt','r',encoding='utf-8')]  
    news_list = []
    for line in f:    
        if line.strip():
            news_cut = list(jieba.cut(''.join(line.strip().split('\t')),cut_all=False,HMM=False))  
            news_list.append([word.strip() for word in news_cut if word not in stop_list and len( word.strip())>0]) 
            # line是：'体育\t马晓旭意外受伤让国奥警惕 无奈大雨格外...'这样的新闻文本，标签是‘体育’，后面是正文，中间用'\t'分开。
            # news_cut : ['体育', '马', '晓', '旭', '意外', '受伤', '让', '国奥', '警惕', ' ', '无奈',...], 按'\t'来拆开
            # news_list为[['体育', '马', '晓', '旭', '意外', '受伤', '国奥', '警惕', '无奈', ...]，去掉了停用词和空格。
    
    news_list = list(chain.from_iterable(news_list))  
    # 原列表中的元素也是列表，把它拉成一个列表。['体育', '马', '晓', '旭', '意外', '受伤', '国奥', '警惕', '无奈', '大雨', ...]
    f.close()
    return news_list

filename = 'data/cnews/cnews.train.txt'
words = read_data(filename)  
# 把所有新闻分词后做成了一个列表：['体育', '马', '晓', '旭', '意外', '受伤', '国奥', '警惕', '无奈', '大雨', ...]

"""第二步：建立词汇表"""
words_size = len(words)      
vocabulary_size = len(set(words))     
print('Data size', vocabulary_size)     
# 共有15457860个，重复的词非常多。
# 词汇表中有196871个不同的词。

def build_dataset(words):
    
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))    
    dictionary = dict()
    # 统计词频较高的词，并得到词的词频。
    # count[:10]: [['UNK', -1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859), ('中国', 55539), ('日', 54018), ('%', 52982), ('基金', 47979), ('更', 40396)]
    #  尽管取了词汇表前（196871-1）个词，但是前面加上了一个用来统计未知词的元素，所以还是196871个词。之所以第一个元素是列表，是为了便于后面进行统计未知词的个数。
    
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # dictionary: {'UNK': 0, '中': 1, '月': 2, '年': 3, '说': 4, '中国': 5,...}，是词汇表中每个字是按照词频进行排序后的，字和它的索引构成的字典。
    
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  
            unk_count += 1
        data.append(index)
        # data是words这个文本列表中每个词对应的索引。元素和words一样多，是15457860个
        # data[:10] : [259, 512, 1023, 3977, 1710, 1413, 12286, 6118, 2417, 18951]
        
    count[0][1] = unk_count       
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))       
    return data, count, dictionary, reverse_dictionary
   # 位置词就是'UNK'本身，所以unk_count是1。[['UNK', 1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859), ('中国', 55539),...]
   # 把字典反转：{0: 'UNK', 1: '中', 2: '月', 3: '年', 4: '说', 5: '中国',...}，用于根据索引取词。

data, count, dictionary, reverse_dictionary = build_dataset(words)
# data[:5] : [259, 512, 1023, 3977, 1710]
# count[:5]: [['UNK', 1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859)]
# reverse_dictionary: {0: 'UNK', 1: '中', 2: '月', 3: '年', 4: '说', 5: '中国',...}

del words        
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# 删掉不同的数据，释放内存。
# Most common words (+UNK) [['UNK', 1], ('中', 96904), ('月', 75567), ('年', 73174), ('说', 56859)]
# Sample data [259, 512, 1023, 3977, 1710, 1413, 12286, 6118, 2417, 18951] ['体育', '马', '晓', '旭', '意外', '受伤', '国奥', '警惕', '无奈', '大雨']

data_index = 0

""" 第三步：为skip-gram模型生成训练的batch """
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)          
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)     
    span = 2 * skip_window + 1  
    buffer = collections.deque(maxlen=span)      
    # 这里先取一个数量为8的batch看看，真正训练时是以128为一个batch的。
    #  构造一个一列有8个元素的ndarray对象
    # deque 是一个双向列表,限制最大长度为5， 可以从两端append和pop数据。
    
    for _ in range(span): 
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)      
        # 循环结束后得到buffer为 deque([259, 512, 1023, 3977, 1710], maxlen=5)，也就是取到了data的前五个值, 对应词语列表的前5个词。
        
    for i in range(batch_size // num_skips):      
        target = skip_window        
        targets_to_avoid = [skip_window] 
        
         # i取值0,1，是表示一个batch能取两个中心词
         # target值为2，意思是中心词在buffer这个列表中的位置是2。
         # 列表是用来存已经取过的词的索引，下次就不能再取了，从而把buffer中5个元素不重复的取完。
         
        for j in range(num_skips):                                                    # j取0，1，2，3，意思是在中心词周围取4个词。
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)                            # 2是中心词的位置，所以j的第一次循环要取到不是2的数字，也就是取到0，1，3，4其中的一个，才能跳出循环。
            targets_to_avoid.append(target)                                       # 把取过的上下文词的索引加进去。
            batch[i * num_skips + j] = buffer[skip_window]               # 取到中心词的索引。前四个元素都是同一个中心词的索引。
            labels[i * num_skips + j, 0] = buffer[target]                     # 取到中心词的上下文词的索引。一共会取到上下各两个。
        buffer.append(data[data_index])                                          # 第一次循环结果为buffer：deque([512, 1023, 3977, 1710, 1413], maxlen=5)，
                                                                                                       # 所以明白了为什么限制为5，因为可以把第一个元素去掉。这也是为什么不用list。
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2)
# batch是 array([1023, 1023, 1023, 1023, 3977, 3977, 3977, 3977], dtype=int32)，8个batch取到了2个中心词，一会看样本的输出结果就明白了。

for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
'''
打印的结果如下，突然明白说为什么说取样本的时候是用bag of words

1023 晓 -> 3977 旭
1023 晓 -> 1710 意外
1023 晓 -> 512 马
1023 晓 -> 259 体育
3977 旭 -> 512 马
3977 旭 -> 1023 晓
3977 旭 -> 1710 意外
3977 旭 -> 1413 受伤

'''

""" 第四步：定义和训练skip-gram模型"""

batch_size = 128            
embedding_size = 300  
skip_window = 2             
num_skips = 4                
num_sampled = 64        
# 上面那个数量为8的batch只是为了展示以下取样的结果，实际上是batch-size 是128。
# 词向量的维度是300维。
# 左右两边各取两个词。
# 要取4个上下文词，同一个中心词也要重复取4次。
# 负采样的负样本数量为64

graph = tf.Graph()         

with graph.as_default():                   
    #  把新生成的图作为整个 tensorflow 运行环境的默认图，详见第二部分的知识点。
    
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])        
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])      
    
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) 
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)    
    #产生-1到1之间的均匀分布, 看作是初始化隐含层和输出层之间的词向量矩阵。
    #用词的索引在词向量矩阵中得到对应的词向量。shape=(128, 300)

    
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    # 初始化损失（loss）函数的权重矩阵和偏置矩阵
    # 生成的值服从具有指定平均值和合理标准偏差的正态分布，如果生成的值大于平均值2个标准偏差则丢弃重新生成。这里是初始化权重矩阵。
    # 对标准方差进行了限制的原因是为了防止神经网络的参数过大。
    
    nce_biases = tf.Variable(tf.zeros([vocabulary_size])) 
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                     labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
    # 初始化偏置矩阵，生成了一个vocabulary_size * 1大小的零矩阵。
    # 这个tf.nn.nce_loss函数把多分类问题变成了正样本和负样本的二分类问题。用的是逻辑回归的交叉熵损失函数来求，而不是softmax  。
    
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))       
    normalized_embeddings = embeddings / norm
    # shape=(196871, 1), 对词向量矩阵进行归一化
    
    init = tf.global_variables_initializer()          
    
num_steps = 10    
with tf.Session(graph=graph) as session:
    
    init.run()
    print('initialized.')
    
    average_loss = 0
  
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        final_embeddings = normalized_embeddings.eval()
        print(final_embeddings)        
        print("*"*20)
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
            
    final_embeddings = normalized_embeddings.eval()      
    # 训练得到最后的词向量矩阵。
    print(final_embeddings)
    fp=open('vector.txt','w',encoding='utf8')
    for k,v in reverse_dictionary.items():
        t=tuple(final_embeddings[k])         
        s=''
        for i in t:
            i=str(i)
            s+=i+" "               
        fp.write(v+" "+s+"\n")         
        # s为'0.031514477 0.059997283 ...'  , 对于每一个词的词向量中的300个数字，用空格把他们连接成字符串。
        #把词向量写入文本文档中。不过这样就成了字符串，我之前试过用np保存为ndarray格式，这里是按源码的保存方式。

    fp.close()

"""第六步：词向量可视化 """
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    myfont = font_manager.FontProperties(fname='/home/dyy/Downloads/font163/simhei.ttf')          #加载中文字体
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),                           #添加注释, xytest是注释的位置。然后添加显示的字体。
                 textcoords='offset points',
                 ha='right',
                 va='bottom',
                 fontproperties=myfont)
    
    plt.savefig(filename)
    plt.show()

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib import font_manager            
    #这个库很重要，因为需要加载字体，原开源代码里是没有的。

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)         
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])           
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]   
    # tsne: 一个降维的方法，降维后维度是2维，使用'pca'来初始化。
    # 取出了前500个词的词向量，把300维减低到2维。
    
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
