# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, learning_curve, train_test_split


"""
逻辑回归预测乳腺癌
"""


# 导入数据，划分训练集和测试集
def get_data():
    cacer = load_breast_cancer()
    x = cacer.data
    y = cacer.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def polynomail_model(degree=1, **kwarg):
    pf = PolynomialFeatures(degree=degree, include_bias=False)

    # penalty='l2', 选择作为正则项的范式
    #   l1范式为向量中各元素的绝对值之和，使得模型中参数为0的维度尽量多，常用于特征选择
    #   l2范式为向量中各元素平方和开根号，使得模型的参数尽量小但不为0，用于正则化
    # C=1.0, 正则化因子（1-αλ/m）中的参数λ，值越小，正则化程度越高
    # class_weight=None, 模型中各类型的权重，通常用来解决两类问题：
    #   1.误分类的代价更高
    #   2.样本高度失衡
    # solver='lbfgs', 对损失函数的优化方法
    #   liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数
    #   lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
    #   newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
    #   sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候
    #   saga：线性收敛的随机优化算法的的变种
    # max_iter=100, 最大迭代次数
    # multi_class='auto',多元回归时，分类方式的选择参数
    lr = LogisticRegression(**kwarg)
    pipe = Pipeline([("polynomial_features", pf), ("logistic_regression", lr)])
    return pipe


# 学习曲线
# train_sizes=np.linspace(0.1,1.0,5)表示从0.1到1分成5等份
def plot_learning_curve(
    plt,
    estimator,
    title,
    x,
    y,
    ylim=None,
    cv=None,
    n_jobs=1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("training sets")
    plt.ylabel("score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # 模型训练的样本是随机分配，每次训练的准确率不一样，选择用均值和方差来描述
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)

    plt.grid()

    # 将均值上下方差的空间用颜色填充
    plt.fill_between(
        train_sizes,
        train_score_mean - train_score_std,
        train_score_mean + train_score_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_score_mean - test_score_std,
        test_score_mean + test_score_std,
        alpha=0.1,
        color="g",
    )

    plt.plot(train_sizes, train_score_mean, "o-", color="r", label="training score")
    plt.plot(train_sizes, test_score_mean, "o-", color="g", label="validation score")

    plt.legend(loc="best")
    return plt


# 分别画出l1范数，l2范数作为正则项的学习曲线的1阶和2阶多项式特征模型的学习曲线
def real_plot(penalty, solver, max_iter, x, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "degree={0},penalty={1}"
    degree = [1, 2]
    plt.figure(figsize=(12, 4), dpi=300)
    for i in range(len(degree)):
        plt.subplot(1, len(degree), i + 1)
        # max_iter 在达到最大迭代次数后，模型收敛程度不够会输出警告信息
        model = polynomail_model(
            degree=degree[i], penalty=penalty, solver=solver, max_iter=max_iter
        )
        plot_learning_curve(
            plt, model, title.format(degree[i], penalty), x, y, ylim=(0.8, 1.01), cv=cv
        )
    plt.savefig("pics/logistic_{}_curve.png".format(penalty))


def run():
    x_train, x_test, y_train, y_test = get_data()
    real_plot("l1", "liblinear", 300, x_train, y_train)
    real_plot("l2", "lbfgs", 1000, x_train, y_train)


if __name__ == "__main__":
    run()
