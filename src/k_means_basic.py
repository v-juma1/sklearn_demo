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
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
"""
使用k-means在生成的数据上聚类
"""

x, y = make_blobs(n_samples=200,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)


#分别使用k=2,3,4 三种不同的聚类个数，并画出聚类后的数据分布图
def k_means_model(n_clusters, x):
    # n_clusters=8, 聚类的个数
    # init='k-means++', 如何选取初始的中心点
    #   kmeans++表示该初始化策略选择的初始均值向量之间都距离比较远，它的效果较好
    #   random表示从数据中随机选择K个样本最为初始均值向量
    #   或者提供一个数组，数组的形状为（n_cluster,n_features），该数组作为初始均值向量
    # n_init=10, 初始化次数
    # max_iter=300, 最大迭代次数
    # tol=1e-4, 算法收敛的阈值
    # precompute_distances='auto', 是否提前计算好样本距离，速度更快，占用内存更多
    # verbose=0, 是否输出日志
    # random_state=None,
    # copy_x=True,
    # n_jobs=None,
    # algorithm='auto' 优化算法的选择，有auto、full和elkan三种选择
    kmean = KMeans(n_clusters=n_clusters)
    kmean.fit(x)

    #k-means 算法的代价是训练样本到其所属聚类中心的距离的平均值
    #在sklearn里面有所不同，score表示的是样本到其所属中心点的和，用负数表示，绝对值越大，成本越高
    score = kmean.score(x)

    labels = kmean.labels_
    centers = kmean.cluster_centers_
    markers = ['o', '^', '*', 's']
    colors = ['r', 'b', 'y', 'k']

    plt.xticks = (())
    plt.yticks = (())
    plt.title("k={},score={}".format(n_clusters, (int)(score)))

    #画出样本
    for c in range(n_clusters):
        cluster = x[labels == c]
        plt.scatter(cluster[:, 0],
                    cluster[:, 1],
                    marker=markers[c],
                    s=20,
                    c=colors[c])

    #画出中心点
    plt.scatter(centers[:, 0],
                centers[:, 1],
                marker='o',
                c="green",
                alpha=0.9,
                s=300)
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, s=50, c=colors[i])


n_clusters = [2, 3, 4]
plt.figure(figsize=(10, 4), dpi=144)
for i, c in enumerate(n_clusters):
    plt.subplot(1, 3, i + 1)
    k_means_model(c, x)

plt.savefig("scikit-learn/log/k-menas_generate_data.png")
