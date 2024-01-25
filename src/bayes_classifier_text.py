# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import *
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

"""
基于tf-idf和贝叶斯进行文本分类
"""


def get_data():
    news_train = load_files(r"data/dataset-379-20news-18828/379/train")
    news_test = load_files(r"data/dataset-379-20news-18828/379/test")

    # 通过计算每个词的tf-idf值并填充到相应位置上，从而将一篇文档转化为向量
    vec = TfidfVectorizer(encoding="latin-1")

    # 每行代表一篇向量化的文档，但这样的矩阵大多数列都是0，因此转化为稀疏矩阵返回
    x_train = vec.fit_transform((d for d in news_train.data))
    x_test = vec.transform((d for d in news_test.data))

    # 使用多项式分布的朴素贝叶斯算法，除此之外，在naive_bayes中还实现了其他分布的贝叶斯算法
    y_train = news_train.target
    y_test = news_test.target
    labels = news_test.target_names
    return x_train, x_test, y_train, y_test, labels


def plot_confusion_matrix(confusion_mat, save_path):
    plt.figure(figsize=(10, 8), dpi=300)
    # 画混淆矩阵图，配色风格使用cm.Greens
    plt.imshow(confusion_mat, interpolation="nearest", cmap=plt.cm.Greens)

    # 显示colorbar
    plt.colorbar()

    # 使用annotate在图中显示混淆矩阵的数据
    for x in range(len(confusion_mat)):
        for y in range(len(confusion_mat)):
            plt.annotate(
                confusion_mat[x, y],
                xy=(x, y),
                horizontalalignment="center",
                verticalalignment="center",
            )
            # 第一个参数是注释的内容
            # xy设置箭头尖的坐标
            # horizontalalignment水平对齐
            # verticalalignment垂直对齐
            # 其余常用参数如下：
            # xytext设置注释内容显示的起始位置
            # arrowprops 用来设置箭头
            # facecolor 设置箭头的颜色
            # headlength 箭头的头的长度
            # headwidth 箭头的宽度
            # width 箭身的宽度

    plt.title("Confusion Matrix")  # 图标title
    plt.ylabel("True label")  # 坐标轴标签
    plt.xlabel("Predicted label")  # 坐标轴标签

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.savefig(save_path)


def fit_model(x_train, x_test, y_train, y_test, labels, name):
    if name == "MultinomialNB":
        # alpha表示平滑系数，值越小，越容易过拟合
        clf = MultinomialNB(alpha=0.0001)
    elif name == "ComplementNB":
        clf = ComplementNB()
    elif name == "BernoulliNB":
        clf = BernoulliNB()

    clf.fit(x_train, y_train)

    # 查看每个类别的预测准确性
    pred = clf.predict(x_test)
    result = classification_report(y_test, pred, target_names=labels)
    logger.info(result)

    # 根据混淆矩阵，查看每个类被误分的情况
    # 其中i行j列的值表示第i类被分为j类的个数
    cm = confusion_matrix(y_test, pred)

    # 绘制 confusion_matrix
    plot_confusion_matrix(cm, "pics/bayes_{}.png".format(name))


def run():
    x_train, x_test, y_train, y_test, labels = get_data()
    fit_model(x_train, x_test, y_train, y_test, labels, "MultinomialNB")
    # fit_model(x_train, x_test, y_train, y_test, labels, "ComplementNB")
    # fit_model(x_train, x_test, y_train, y_test, labels, "BernoulliNB")


if __name__ == "__main__":
    run()
