# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    SGDRegressor,
    Lasso,
    ElasticNet,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.figure import SubplotParams

"""
使用线性回归模拟正弦函数
"""


# 画出学习曲线，模型在验证集和训练集上均方根误差随着训练数据大小的变化情况
def plot_learning_curves(models, X_train, X_val, y_train, y_val, pic_path):
    # 绘制4个子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 遍历每个模型并绘制RMSE随着训练数据增长的变化情况
    for i, m in enumerate(models):
        model = m["model"]
        params = m["params"]

        row = i // 2
        col = i % 2
        axs[row, col].set_title(params)
        train_errors, val_errors = [], []
        for m in range(1, len(X_train) + 1):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))
        axs[row, col].plot(np.sqrt(train_errors), "g", label="train")
        axs[row, col].plot(np.sqrt(val_errors), "b", label="val")
        axs[row, col].legend(loc="upper right", fontsize=14)
        axs[row, col].set_xlabel("Training set size")
        axs[row, col].set_ylabel("RMSE")

    # 调整子图之间的间距和位置
    fig.tight_layout()
    # 保存
    plt.savefig(pic_path)


# 画出4个模型的拟合效果
def draw_performance(models, x, y, pic_path):
    plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
    for i, model in enumerate(models):
        fig = plt.subplot(2, 2, i + 1)
        plt.xlim(-8, 8)
        plt.title(model["params"])
        plt.scatter(x, y, s=5, c="b", alpha=0.5)
        plt.plot(x, model["model"].predict(x), "r--")

    plt.savefig(pic_path)


# 生成数据
def generate_data(n_dots=300):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
    y = np.sin(x) + 0.5 * np.random.rand(n_dots) - 0.5
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y


# 多项式回归
def polynomial_model():
    # 4个不同阶数的多项式回归模型
    models = []
    # 2,3,5,10阶的多项式特征
    for degree in [2, 3, 5, 10]:
        pf = PolynomialFeatures(degree=degree)
        lr = LinearRegression()
        pipe = Pipeline([("polynomial_feature", pf), ("linear_regression", lr)])
        models.append({"model": pipe, "params": "degree={}".format(degree)})
    return models


# 岭回归
def ridge_model():
    models = []
    for alpha in [0, 0.5, 1, 10]:
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=10)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", Ridge(alpha=alpha, solver="cholesky")),
            ]
        )

        models.append({"model": model, "params": "alpha={}".format(alpha)})

    return models


def sgd_model():
    models = []
    for alpha in [0, 0.1, 0.001, 0.00001]:
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=10)),
                ("std_scaler", StandardScaler()),
                ("regul_sgd", SGDRegressor(penalty="l2", alpha=alpha)),
            ]
        )

        models.append({"model": model, "params": "alpha={}".format(alpha)})

    return models


def Lasso_model():
    models = []
    for alpha in [0.1, 0.05, 0.001, 0]:
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=10)),
                ("std_scaler", StandardScaler()),
                ("regul_Lasso", Lasso(alpha=alpha)),
            ]
        )

        models.append({"model": model, "params": "alpha={}".format(alpha)})

    return models


def ElasticNet_model():
    models = []
    for l1_ratio in [0, 0.25, 0.75, 1]:
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=10)),
                ("std_scaler", StandardScaler()),
                ("regul_ElasticNet", ElasticNet(alpha=0.1, l1_ratio=l1_ratio)),
            ]
        )

        models.append({"model": model, "params": "l1_ratio={}".format(l1_ratio)})

    return models


# fit 4个 model 并且返回mse
def fit_model(models, x, y):
    for model in models:
        model["model"].fit(x, y)
        train_score = model["model"].score(x, y)
        # 均方根误差，实际的点和模型预测的点之间的距离,值越小，模型越好
        mse = mean_squared_error(y, model["model"].predict(x))
        logger.info("model:{}".format(model["model"]))
        logger.info("params:{}".format(model["params"]))
        logger.info("train_score:{}".format(train_score))
        logger.info("mse:{}".format(mse))


def run(model_name, task):
    # 生成数据
    x, y = generate_data()
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if model_name == "polynomial":
        models = polynomial_model()
    elif model_name == "ridge":
        models = ridge_model()
    elif model_name == "sgd":
        models = sgd_model()
    elif model_name == "Lasso":
        models = Lasso_model()
    elif model_name == "ElasticNet":
        models = ElasticNet_model()

    if task == "performance":
        # 每次训练四个模型
        fit_model(models, xtrain, ytrain)

        # 画出四个模型的拟合情况
        draw_performance(models, x, y, pic_path="pics/{}_sin.png".format(model_name))
    elif task == "learning_curve":
        plot_learning_curves(
            models,
            xtrain,
            xtest,
            ytrain,
            ytest,
            pic_path="pics/{}_learning_curve.png".format(model_name),
        )


if __name__ == "__main__":
    # run("polynomial", "performance")
    # run("polynomial", "learning_curve")
    # run("ridge", "performance")
    # run("ridge", "learning_curve")
    # run("sgd", "performance")
    # run("sgd", "learning_curve")
    # run("Lasso", "performance")
    # run("Lasso", "learning_curve")
    run("ElasticNet", "performance")
    run("ElasticNet", "learning_curve")
