# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    return X, y


# 梯度boost，在前一个模型的残差上训练
def fit_scratc_model(X, y):
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)

    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X, y2)

    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X, y3)

    X_new = np.array([[0.8]])
    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    logger.info(y_pred)


def fit_model(X, y):
    gbrt_slow = GradientBoostingRegressor(
        max_depth=2, n_estimators=100, learning_rate=0.1, random_state=42
    )
    gbrt_slow.fit(X, y)
    X_new = np.array([[0.8]])
    y_pred = gbrt_slow.predict(X_new)
    logger.info(y_pred)


def run():
    X, y = get_data()

    fit_scratc_model(X, y)
    fit_model(X, y)


if __name__ == "__main__":
    run()
