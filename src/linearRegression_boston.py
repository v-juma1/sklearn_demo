# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import logging.config
#读取日志配置文件
logging.config.fileConfig("scikit-learn/conf/logging.conf")

#选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
"""
sklearn 自带的波士顿房价数据集,13维特征
"""

boston = load_boston()
x = boston.data
y = boston.target

#%20的数据作为测试数据
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=3)


#多项式特征
def polynomial_feature(degree=1):
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    lr = LinearRegression(normalize=True)
    pipe = Pipeline([("polynomial_features", pf), ("linear_regression", lr)])
    return pipe


model = polynomial_feature(degree=2)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
logger.info(score)
