# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import dtreeviz


"""
sepal length (cm):花萼长度,
sepal width (cm):花萼宽度,
petal length (cm):花瓣长度,
petal width (cm):花瓣宽度,

["setosa:山鸢尾", "versicolor:变色鸢尾", "virginica:维吉尼亚鸢尾"]

"""


def load_data():
    iris = load_iris()
    # features = list(iris.feature_names)
    # class_names = iris.target_names

    X = iris.data
    y = iris.target

    return X, y


def fit_classifier(X, y):
    features = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
    ]
    class_names = ["setosa", "versicolor", "virginica"]
    iris_classifier = DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=1, random_state=42
    )
    iris_classifier.fit(X, y)

    # 可视化
    viz_model = dtreeviz.model(
        iris_classifier,
        X_train=X,
        y_train=y,
        feature_names=features,
        target_name="iris",
        class_names=class_names,
    )

    viz_model.view(scale=0.8).save("pics/classifier_iris.svg")

    # 预测或者输出概率
    one_data = [7.0, 3.2, 4.7, 1.4]
    logger.info(iris_classifier.predict([one_data]))

    # 可视化一条数据被分类的过程
    viz_model.view(scale=0.8, x=one_data).save("pics/classifier_one_iris.svg")

    # 可视化一条数据被分类的过程，只画出分类过程中用到的分支，其他分支忽略
    viz_model.view(scale=0.8, x=one_data, show_just_path=True).save(
        "pics/classifier_onepath_iris.svg"
    )

    # 现在分类树是最大生长，没有剪枝，所以输出的概率数组中，只有一个为1，其他为0
    # 剪枝（如设置max_depth）以后，概率才会分布到数组中其他位置
    logger.info(iris_classifier.predict_proba([[7.0, 3.2, 4.7, 1.4]]))


def fit_regressor(X, y):
    features = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
    ]
    class_names = ["setosa", "versicolor", "virginica"]
    iris_regressor = DecisionTreeRegressor(
        max_depth=None, min_samples_leaf=1, random_state=42
    )
    iris_regressor.fit(X, y)

    # 可视化
    viz_model = dtreeviz.model(
        iris_regressor,
        X_train=X,
        y_train=y,
        feature_names=features,
        target_name="iris",
        class_names=class_names,
    )

    viz_model.view(scale=0.8).save("pics/regressor_iris.svg")

    # 预测
    logger.info(iris_regressor.predict([[7.0, 3.2, 4.7, 1.4]]))


def run():
    X, y = load_data()

    fit_classifier(X, y)
    fit_regressor(X, y)


if __name__ == "__main__":
    run()
