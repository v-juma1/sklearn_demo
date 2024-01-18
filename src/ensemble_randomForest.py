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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

random_state = 42


def get_data():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def fit_predict_model(X_train, X_test, y_train, y_test):
    # 500 个决策树组成的随机森林
    # 和bagging的方法相比，在树的生长上引入更多随机性
    # 分裂节点时不再搜索最好的特征，而是在一个随机生成的特征子集里搜索最好的特征
    rnd_clf = RandomForestClassifier(
        n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42
    )

    rnd_clf.fit(X_train, y_train)

    # 输出每个属性的重要性
    logger.info("{}".format(rnd_clf.feature_importances_))

    # 输出测试集上的结果
    y_pred = rnd_clf.predict(X_test)
    logger.info(" RandomForest_test_score: {}".format(accuracy_score(y_test, y_pred)))

    # 跟单个决策树相比，可以看到结果有提升
    tree_clf = DecisionTreeClassifier(
        max_leaf_nodes=16,
        random_state=42,
    )
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    logger.info("single_tree_score: {}".format(accuracy_score(y_test, y_pred_tree)))


def run():
    X_train, X_test, y_train, y_test = get_data()
    fit_predict_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run()
