# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def generate_data(random_state=42):
    np.random.seed(random_state)
    m = 100
    X = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1) / 10).ravel()
    return X, y


def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors="#FFAAAA")
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


def plot_regression(X, y):
    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1, gamma="scale")
    svm_poly_reg1.fit(X, y)
    svm_poly_reg2.fit(X, y)

    fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
    plt.sca(axes[0])
    plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
    plt.title(
        r"$degree={}, C={}, \epsilon = {}$".format(
            svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon
        ),
        fontsize=18,
    )
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.sca(axes[1])
    plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
    plt.title(
        r"$degree={}, C={}, \epsilon = {}$".format(
            svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon
        ),
        fontsize=18,
    )
    plt.savefig("pics/svm_polynomial_kernel.png")


def run():
    X, y = generate_data()
    plot_regression(X, y)


if __name__ == "__main__":
    run()
