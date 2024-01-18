# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

random_state = 42


def get_data():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def fit_predict_model(X_train, X_test, y_train, y_test):
    # 500 个决策树组成的Bagging 模型
    # 同质的模型，通过在 Bootstraping 方法上得的出采样集训练出500个决策树
    # Bootstraping采样方法会剩余大概37%的未被采样的实例，即oob数据
    # 设置oob_score参数可以可以在这些实例上进行评估

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=500,
        max_samples=100,
        bootstrap=True,
        random_state=42,
        oob_score=True,
    )

    bag_clf.fit(X_train, y_train)

    # 输出在oob数据上的结果
    logger.info("bag_oob_score: {}".format(bag_clf.oob_score_))

    # 输出测试集上的结果
    y_pred = bag_clf.predict(X_test)
    logger.info("bag_test_score: {}".format(accuracy_score(y_test, y_pred)))

    # 跟单个决策树相比，可以看到结果有提升
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    logger.info("single_tree_score: {}".format(accuracy_score(y_test, y_pred_tree)))

    # 单个决策树和bagging方法的结果，画个图比较一下：
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision Tree", fontsize=14)
    plt.sca(axes[1])
    plot_decision_boundary(bag_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)
    plt.ylabel("")
    plt.savefig("pics/DecisionTrees_Bagging.png")


def plot_decision_boundary(
    clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True
):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(["#7d7d58", "#4c4c7f", "#507d50"])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


def run():
    X_train, X_test, y_train, y_test = get_data()
    fit_predict_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run()
