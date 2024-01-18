# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

import pandas as pd
import numpy as np
import joblib
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
    GridSearchCV,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

"""
longitude  经度
latitude  纬度
housing_median_age  房龄中位数
total_rooms  房间总数
total_bedrooms 卧室数 
population  人口数
households  家庭数
median_income  收入中位数
median_house_value 房价中位数 (预测的目标,也就是y)
ocean_proximity 房子靠海的情况

"""


# 加载数据
def load_house_data(path):
    return pd.read_csv(path)


# 分层采样的方法划分数据集合
# 用corr()方法计算或者用 scatter_matrix 画出各个属性和预测目标【median_house_value 房价中位数】之间的标准相关系数
# 观察数据中各个属性和预测目标之间的相关性，选择用【median_income  收入中位数】 分层按比例采样
def split_data_by_median_income(data, test_size, seed):
    # 把收入分为5个层级
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    # 将派生出来的列删除
    train_set = strat_train_set.drop(["income_cat"], axis=1)
    test_set = strat_test_set.drop(["income_cat"], axis=1)

    # 【median_house_value 房价中位数】 作为 y,剩下的列作为特征 x
    train_label = train_set["median_house_value"].copy()

    tain_data = train_set.drop(["median_house_value"], axis=1)

    test_label = test_set["median_house_value"].copy()
    test_data = test_set.drop(["median_house_value"], axis=1)

    return tain_data, train_label, test_data, test_label


# 根据已有的属性生成组合的属性
class CombineAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]

        population_per_household = X[:, 5] / X[:, 6]
        bedrooms_per_room = X[:, 4] / X[:, 3]

        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]


def clean_house_data(data):
    # 处理数值类型数据: 填充、组合、缩放
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attributes_add", CombineAttributeAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_attribute = ["ocean_proximity"]
    num_attribute = data.columns.tolist()
    num_attribute.remove("ocean_proximity")
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribute),
            ("cat", OneHotEncoder(sparse_output=False), cat_attribute),
        ]
    )

    clean_data = full_pipeline.fit_transform(data)
    return clean_data


# 用GridSearchCV寻找最佳参数
# 用cross_val_score查看模型性能
def fine_tuning(model, param, datax, datay):
    # 用GridSearchCV寻找最优参数，fine-tuning
    grid_search = GridSearchCV(
        model, param, cv=5, scoring="neg_mean_squared_error", verbose=3
    )
    grid_search.fit(datax, datay)

    # 最佳参数
    logger.info("最佳参数:{%s}", grid_search.best_params_)

    # 使用k折交叉验证更好的评估模型,看看最好的参数对应的模型表现如何
    # 将数据随机分成10份，1份用于测试，9份用于训练
    # 训练10次，返回一个包含10个结果的数组
    neg_score = cross_val_score(
        grid_search.best_estimator_,
        datax,
        datay,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=3,
    )
    tree_rmse_scores = np.sqrt(-neg_score)

    logger.info("最佳参数模型mean_score:{%s}", tree_rmse_scores.mean())
    return grid_search.best_params_


class house_model:
    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, model, data, label):
        model.fit(data, label)

        tree_predictions = model.predict(data)
        tree_mse = mean_squared_error(label, tree_predictions)
        tree_rmse = np.sqrt(tree_mse)
        logger.info("MSE:%f, rmse:%f", tree_mse, tree_rmse)

        # 保存模型
        joblib.dump(model, self.model_path)

    def test(self, data, label):
        model = joblib.load(self.model_path)
        y_pred = model.predict(data)
        tree_mse = mean_squared_error(label, y_pred)
        tree_rmse = np.sqrt(tree_mse)
        logger.info("MSE:%f, rmse:%f", tree_mse, tree_rmse)

        # 输出置信区间
        confidence = 0.95
        squared_error = (y_pred - label) ** 2
        Confidence_interval = np.sqrt(
            stats.t.interval(
                confidence,
                len(squared_error) - 1,
                loc=squared_error.mean(),
                scale=stats.sem(squared_error),
            )
        )

        logger.info("Confidence_interval:%s", Confidence_interval)

        # 打印部分结果
        logger.info("true_label vs predicted_label")
        for i, j in zip(label[0:10], y_pred[0:10]):
            logger.info("{%s},{%s}", i, j)

    def predict(self, data):
        model = joblib.load(self.model_path)
        y_pred = model.predict(data)
        logger.info("predicted_label:{%s}", y_pred)
        return y_pred


def run(model_name, task):
    # 加载数据
    data = load_house_data("data/housing/housing.csv")

    # 划分数据集
    tain_data, train_label, test_data, test_label = split_data_by_median_income(
        data, 0.3, 42
    )

    clean_train_data = clean_house_data(tain_data)
    clean_test_data = clean_house_data(test_data)

    if model_name == "RandomForest":
        pre_model = RandomForestRegressor()
        grid_params = [
            # 尝试 12 (3×4) 种参数组合
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        ]
        model_path = "model/house_RandomForestRegressor.pkl"

    elif model_name == "LinearRegression":
        # 使用Pipeline对象时，需要给每个步骤指定一个名称
        polynomial_features = PolynomialFeatures()
        linear_regression = LinearRegression()
        pre_model = Pipeline([("pf", polynomial_features), ("lr", linear_regression)])

        # 设置GridSearchCV时，使用双下划线将 参数名称与 步骤名分隔开来
        grid_params = {"pf__degree": [1, 2, 3], "lr__fit_intercept": [True, False]}

        model_path = "model/house_LinearRegression.pkl"

    model = house_model(model_path)

    if task == "train":
        best_params = fine_tuning(pre_model, grid_params, clean_train_data, train_label)
        pre_model.set_params(**best_params)
        model.train(pre_model, clean_train_data, train_label)

    elif task == "test":
        model.test(clean_test_data, test_label)

    elif task == "predict":
        model.predict(clean_test_data[0:10])


if __name__ == "__main__":
    run("RandomForest", "train")
    # run("RandomForest", "test")
    # run("RandomForest", "predict")

    # run("LinearRegression", "train")
    # run("LinearRegression", "test")
    # run("LinearRegression", "predict")
