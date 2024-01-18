# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

"""
使用k_邻近对连续区间内的数值进行回归拟合
"""


# 生成数据
def get_data():
    n_dots = 200
    x = 5 * np.random.rand(n_dots, 1)
    y = np.cos(x).ravel()
    # 噪声点
    y += 0.2 * np.random.rand(n_dots) - 0.1

    return x, y


def fit_model(x, y, k=5):
    knn = KNeighborsRegressor(k)
    knn.fit(x, y)

    # 在x轴上指定区间生成足够多的点，使用模型预测这些点的y值，然后将所有预测点连接起来形成曲线
    t = np.linspace(0, 5, 1000)[:, np.newaxis]
    y_pred = knn.predict(t)

    plt.figure(figsize=(16, 10), dpi=144)
    plt.scatter(x, y, c="g", label="data", s=100)
    plt.plot(t, y_pred, c="k", label="prediccion", lw=4)
    plt.axis("tight")
    plt.title("KNeighborsRegressor")
    plt.savefig("pics/KNeighborsRegresso_k={}.png".format(k))


def run():
    x, y = get_data()
    fit_model(x, y, k=5)
    fit_model(x, y, k=50)


if __name__ == "__main__":
    run()
