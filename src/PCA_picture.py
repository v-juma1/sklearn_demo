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
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
"""
使用PCA压缩图片并分类
"""

#fetch_olivetti_faces裁剪原图片，只保留脸部且居中
#数据维度(400, 4096),特征有4096维
faces = fetch_olivetti_faces(r"scikit-learn/data/pic")
x = faces.data
y = faces.target

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=4)

#选择前k个重要特征
#根据还原率和k的关系选择最优的k


def best_k(x):
    ks = range(10, 300, 30)
    rate = []

    for c in ks:
        pca = PCA(n_components=c)
        x_pca = pca.fit_transform(x)
        #explained_variance_ratio_获取处理后数据的还原率
        rate.append(np.sum(pca.explained_variance_ratio_))

    #画出还原率和k之间的关系图
    plt.figure(figsize=(10, 6), dpi=144)
    plt.grid()
    plt.plot(ks, rate)
    plt.xlabel("还原率")
    plt.ylabel("k")
    plt.xticks(np.arange(0, 300, 20))
    plt.yticks(np.arange(0.5, 1.05, 0.05))

    plt.savefig("scikit-learn/log/pca_best_k.png")


#从上面函数的图中可以看出，还原率在95%以上时，最优的k大概为140
#使用这个参数对数据进行降维
pca = PCA(n_components=140, svd_solver='randomized', whiten=True).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

#使用svm进行分类，并用GridSearchCV寻找最优的参数组合
params = {
    'C': [1, 5, 10, 50, 100],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]
}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   params,
                   verbose=2,
                   n_jobs=4)
clf.fit(x_train_pca, y_train)

#使用最优点的模型进行预测，并输出分类结果
y_pred = clf.best_estimator_.predict(x_test_pca)
logger.info(classification_report(y_test, y_pred))
