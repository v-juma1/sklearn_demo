# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import logging.config
#读取日志配置文件
logging.config.fileConfig("scikit-learn/conf/logging.conf", encoding="utf8")

#选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def run():
    pass


if __name__ == "__main__":
    run()