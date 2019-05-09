#encoding=utf8
import jieba
from gensim.models import word2vec
import gensim

# 第一步：加入字典
def add_dict():
    f = open('./text/special_nouns.txt','r', encoding='utf-8')
    for word in f:
        jieba.suggest_freq(word.strip(),tune=True)
    f.close()
    
add_dict()

#  第二步：读取三体小说的文本，并进行分词
def document_segment(filename):
    f = open(filename, 'r',encoding='utf-8')
    document = f.read()
    document_cut = ' '.join(jieba.cut(document))      
    with open('./text/The_three_body_problem_segment.txt','w',encoding='utf-8') as f2:
        f2.write(document_cut)     # 
    f.close()
    f2.close()
    
document_segment('./text/The_three_body_problem.txt')
    
# 第三步：训练词向量和保存模型

def train_w2v(filename):
    
    text = word2vec.LineSentence(filename)
    model = word2vec.Word2Vec(text, sg = 0, hs=1,min_count=1,window=3,size=100)
    model.save('./my_model')
    
train_w2v('./text/The_three_body_problem_segment.txt')

model = word2vec.Word2Vec.load('./my_model')

# 取出词语对应的词向量。
vec = model[['红岸','水滴','思想钢印']]
print('三个词的词向量矩阵的维度是：', vec.shape,'。')
print('-------------------------------我是分隔符------------------------')
# 计算两个词的相似程度。
print('叶文洁和红岸的余弦相似度是：', model.similarity('叶文洁', '红岸'),'。')
print('-------------------------------我是分隔符------------------------')
# 得到和某个词比较相关的词的列表
sim1 = model.most_similar('叶文洁',topn=10)
for key in sim1:
    print('和叶文洁比较相关的词有',key[0],'，余弦距离是：',key[1])
    
