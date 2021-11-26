# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import logging.config
#读取日志配置文件
logging.config.fileConfig("scikit-learn/conf/logging.conf")

#选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
"""
使用k-means进行文本聚类
"""


def text_cluster():
    docs = load_files(r"scikit-learn/data/clustering/data")

    # input='content', 输入的类型，文件名(filename),可读对象（file）,文本内容（contnet）
    # encoding='utf-8',
    # decode_error='strict', 编码错误时如何处理
    # strip_accents=None,
    # lowercase=True,
    # preprocessor=None,
    # tokenizer=None,
    # analyzer='word', 向量化的单位是字还是词
    # stop_words=None, 停用词表
    # token_pattern=r"(?u)\b\w\w+\b", 匹配一个token(字或者词)的正则
    # ngram_range=(1, 1),
    # max_df=1.0, 在百分之多少中出现的词，被认为是高频词，从而被剔除
    # min_df=1, 只在多少篇文档中出现的词，被认为是低频词，也被剔除
    # max_features=None, 根据tf-idf的大小从大到小排序，取前max_features个词构成词典,该值即为最后一篇文档的向量维数
    # vocabulary=None, 词汇表，如果没有给出，则从文档中确定
    # binary=False,
    # dtype=np.float64, 返回的矩阵类型
    # norm='l2', 正则化向量的方法
    # use_idf=True, 是否启用idf
    # smooth_idf=True, 对idf进行加一平滑，防止分母为零
    # sublinear_tf=False 是否对tf进行缩放，即用1 + log（tf）替换tf
    vector = TfidfVectorizer(max_df=0.4,
                             min_df=2,
                             max_features=20000,
                             encoding="latin-1")
    x = vector.fit_transform((d for d in docs.data))

    #聚成4类，最大迭代次数100，距中心点距离效于0.1即认为已经收敛，进行3次k均值运算后求平均值确定初始中心点
    n_clusters = 4
    kmean = KMeans(n_clusters=n_clusters, max_iter=100, tol=0.1, n_init=3)
    kmean.fit(x)

    #查看聚类过程中，每类文档权限最高的10个词
    #argsort()函数把一个numpy数组按照从小到大排序，并返回索引，::-1则是把升序改为降序
    #vector.get_feature_names()得到词典的单词
    index = kmean.cluster_centers_.argsort()[:, ::-1]
    terms = vector.get_feature_names()
    for i in range(n_clusters):
        print("cluster %d:" % i, end=" ")
        for ind in index[i, :10]:
            print(' %s' % terms[ind], end='')

        print()

    #对无监督的聚类算法性能进行评估
    #轮廓系数：b-a/max(a,b)
    #   a:一个样本与其相同聚类的平均距离
    #   b:一个样本与其距离最近的下一个聚类里的点的平均距离
    #值在[-1,1]之间，-1表示完全错误的分类，0表示分类有重叠，1表示完美的聚类

    print(metrics.silhouette_score(x, kmean.labels_,
                                   sample_size=1000))  #0.0052792296749458855


#评价聚类算法的指标
#这些指标都只适用于已经标记的类别数据，即用标记的数据，采用有监督的学习方法
def cluster_evaluate():
    #Adjust Rand Index ，衡量两个序列相似性的算法，越不同的序列，值接近负数或者0，越相同越接近1
    true_label = np.random.randint(1, 4, 6)
    pred_label = np.random.randint(1, 4, 6)

    print(metrics.adjusted_rand_score(true_label,
                                      pred_label))  #-0.17647058823529416

    true_label = [1, 1, 3, 3, 2, 2]
    pred_label = [3, 3, 2, 2, 1, 1]

    print(metrics.adjusted_rand_score(true_label, pred_label))  #1.0

    #齐次性：一个聚类元素只能由一种类别的元素组成
    true_label = [1, 1, 2, 2]
    pred_label = [2, 2, 1, 1]
    print(metrics.homogeneity_score(true_label, pred_label))  #1.0

    true_label = [1, 1, 2, 2]
    pred_label = [0, 1, 2, 3]
    #这个例子中，标记了2个聚类，输出了四个聚类且包含已标记的聚类
    #每个已标记的聚类由一个输出类别组成，因此齐次性条件满足
    print(metrics.homogeneity_score(true_label,
                                    pred_label))  #0.9999999999999999

    true_label = [1, 1, 2, 2]
    pred_label = [1, 2, 1, 2]
    #其中1和2两个类都是由输出中的两个类别组成，因此不满足齐次性条件
    print(metrics.homogeneity_score(true_label, pred_label))  #0.0

    true_label = np.random.randint(1, 4, 6)
    pred_label = np.random.randint(1, 4, 6)
    #对随机序列，齐次性值不为0
    print(metrics.homogeneity_score(true_label,
                                    pred_label))  #0.20751874963942177

    #完整性：给定已标记的类别，全部分配到一个聚类中
    true_label = [1, 1, 2, 2]
    pred_label = [2, 2, 1, 1]
    print(metrics.completeness_score(true_label, pred_label))  #1.0

    true_label = [0, 1, 2, 3]
    pred_label = [1, 1, 2, 2]
    print(metrics.completeness_score(true_label,
                                     pred_label))  #0.9999999999999999

    true_label = [1, 1, 2, 2]
    pred_label = [1, 2, 1, 2]
    #1和2两个类别都被分到了不同的聚类中
    print(metrics.completeness_score(true_label, pred_label))  #0.0

    true_label = np.random.randint(1, 4, 6)
    pred_label = np.random.randint(1, 4, 6)
    print(metrics.completeness_score(true_label,
                                     pred_label))  #0.41102631318192145

    #将两个指数合起来，成为V-measure分数
    true_label = [1, 1, 2, 2]
    pred_label = [2, 2, 1, 1]
    #齐次性和完整性都满足
    print(metrics.v_measure_score(true_label, pred_label))  #1.0

    true_label = [0, 1, 2, 3]
    pred_label = [1, 1, 2, 2]
    #只满足完整性
    print(metrics.v_measure_score(true_label, pred_label))  #0.6666666666666666

    true_label = [1, 1, 2, 2]
    pred_label = [1, 2, 1, 2]
    #齐次性和完整性都  不  满足
    print(metrics.v_measure_score(true_label, pred_label))  #0.0


text_cluster()