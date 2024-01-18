# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


"""
用决策树来预测泰坦尼克号的幸存者
"""


# 读取数据集,并进行预处理
def read_dataset(fname):
    # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用的数据
    data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    # 处理性别数据
    data["Sex"] = (data["Sex"] == "male").astype("int")
    # 处理登船港口数据
    labels = data["Embarked"].unique().tolist()
    data["Embarked"] = data["Embarked"].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)

    y = data["Survived"].values
    x = data.drop(["Survived"], axis=1).values
    return x, y


# 模型的优化方法，选择某个参数，在一组范围确定这个参数的最优值
# 比如max_depth（模型深度）
# 找出得分最高的max_depth，并画出其和模型评分的关系
def max_depth_score(x_train, x_test, y_train, y_test):
    def cv_score(d):
        clf = DecisionTreeClassifier(max_depth=d)
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        return (train_score, test_score)

    depths = range(2, 15)
    scores = [cv_score(d) for d in depths]
    train_scores = [s[0] for s in scores]
    test_scores = [s[1] for s in scores]

    # 找出测试集上score最高的max_depth
    best_score_index = np.argmax(test_scores)
    best_score = test_scores[best_score_index]
    best_param = depths[best_score_index]
    # print("depth:{},score:{}".format(best_param,best_score)

    # 画出关系图
    plt.figure(figsize=(6, 4), dpi=144)
    plt.grid()
    plt.xlabel("max depth of decision tree")
    plt.ylabel("score")
    plt.plot(depths, train_scores, ".g-", label="train score")
    plt.plot(depths, test_scores, ".r--", label="test score")
    plt.legend()
    plt.savefig("pics/max_depth_score.png")


# 找出得分最高的min_impurity_decrease（信息增益的阈值），并画出其和模型评分的关系
def min_impurity_decrease_score(x_train, x_test, y_train, y_test):
    def cv_score(val):
        clf = DecisionTreeClassifier(criterion="gini", min_impurity_decrease=val)
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        return (train_score, test_score)

    values = np.linspace(0, 0.5, 50)
    scores = [cv_score(d) for d in values]
    train_scores = [s[0] for s in scores]
    test_scores = [s[1] for s in scores]

    # 找出测试集上score最高的max_depth
    best_score_index = np.argmax(test_scores)
    best_score = test_scores[best_score_index]
    best_param = values[best_score_index]
    # print("min_impurity_decrease:{},score:{}".format(best_param,best_score)

    # 画出关系图
    plt.figure(figsize=(6, 4), dpi=144)
    plt.grid()
    plt.xlabel("min_impurity_decrease of decision tree")
    plt.ylabel("score")
    plt.plot(values, train_scores, ".g-", label="train score")
    plt.plot(values, test_scores, ".r--", label="test score")
    plt.legend()
    plt.savefig("pics/min_impurity_decrease_score.png")


# 以上挨个寻找最优参数的方法有两个问题
# 1. 数据不稳定
# 2. 不能一次选择多个参数最优的参数组合
# 使用sklean的模型参数选择工具包GridSearchCV
def GridSearchCV_curve(x_train, x_test, y_train, y_test):
    # 画出模型参数于模型评分的关系图
    def plot_curve(train_size, cv_result, xlabel):
        mean_train = cv_result["mean_train_score"]
        std_train = cv_result["std_train_score"]
        mean_test = cv_result["mean_test_score"]
        std_test = cv_result["std_test_score"]

        plt.figure(figsize=(6, 4), dpi=144)
        plt.title("parameters turning")
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel("score")

        # 填充上下两个方差的区间
        plt.fill_between(
            train_size,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.1,
            color="r",
        )

        plt.fill_between(
            train_size, mean_test - std_test, mean_test + std_test, alpha=0.1, color="g"
        )

        plt.plot(
            train_size,
            cv_result["mean_train_score"],
            ".--",
            color="r",
            label="train score",
        )
        plt.plot(
            train_size,
            cv_result["mean_test_score"],
            ".-",
            color="g",
            label="test score",
        )
        plt.legend(loc="best")
        plt.savefig("pics/GridSearchCV_curve.png")

    space = np.linspace(0, 0.5, 50)

    params = {"min_impurity_decrease": space}

    # GridSearchCV会根据params中的值来构建模型，其中并计算平均分和标准差，cv=5表示每次计算把数据分为5份，一份测试，其他用来训练
    # 得到的最优参数及评分保存在clf.best_params_和clf.best_score_中，clf.cv_results_保存了计算过程中的中间结果
    clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5, return_train_score=True)
    clf.fit(x_train, y_train)
    print("params:{0}\n score:{1}".format(clf.best_params_, clf.best_score_))

    plot_curve(space, clf.cv_results_, xlabel="gini thresholds")


def run():
    x, y = read_dataset("data/titanic/train.csv")
    # 分割数据x
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    max_depth_score(x_train, x_test, y_train, y_test)

    min_impurity_decrease_score(x_train, x_test, y_train, y_test)

    GridSearchCV_curve(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    run()
