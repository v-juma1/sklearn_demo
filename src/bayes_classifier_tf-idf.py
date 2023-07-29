# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import logging.config
#读取日志配置文件
logging.config.fileConfig("scikit-learn/conf/logging.conf", encoding="utf8")

#选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
"""
基于tf-idf和朴素贝叶斯进行文本分类
"""


def get_data():
    news_train = load_files(
        r"scikit-learn/data/dataset-379-20news-18828/379/train")
    news_test = load_files(
        r"scikit-learn/data/dataset-379-20news-18828/379/test")

    #通过计算每个词的tf-idf值并填充到相应位置上，从而将一篇文档转化为向量
    vec = TfidfVectorizer(encoding="latin-1")

    #每行代表一篇向量化的文档，但这样的矩阵大多数列都是0，因此转化为稀疏矩阵返回
    x_train = vec.fit_transform((d for d in news_train.data))
    x_test = vec.transform((d for d in news_test.data))

    #使用多项式分布的朴素贝叶斯算法，除此之外，在naive_bayes中还实现了其他分布的贝叶斯算法
    y_train = news_train.target
    y_test = news_test.target
    labels = news_test.target_names
    return x_train, x_test, y_train, y_test, labels


def fit_model(x_train, x_test, y_train, y_test, labels):

    #alpha表示平滑系数，值越小，越容易过拟合
    clf = MultinomialNB(alpha=0.0001)
    clf.fit(x_train, y_train)

    #查看每个类别的预测准确性
    pred = clf.predict(x_test)
    result = classification_report(y_test, pred, target_names=labels)
    logger.info(result)

    #根据混淆矩阵，查看每个累被误分的情况
    #其中i行j列的值表示第i类被分为j类的个数
    cm = confusion_matrix(y_test, pred)
    logger.info(cm)


def run():
    x_train, x_test, y_train, y_test, labels = get_data()
    fit_model(x_train, x_test, y_train, y_test, labels)


if __name__ == "__main__":
    run()