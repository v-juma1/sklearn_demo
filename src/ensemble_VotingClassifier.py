# -*- coding: utf-8 -*-

import logging
import logging.config as log_config

# 读取日志配置文件
log_config.fileConfig("conf/logging.conf", encoding="utf8")

# 选择配置在[loggers]中的选项
logger = logging.getLogger("fileAndConsole")

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

random_state = 42


def get_data():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test


def fit_predict_model(X_train, X_test, y_train, y_test):
    # 三个异质的模型（模型的算法和超参数都不同）
    log_clf = LogisticRegression(solver="lbfgs", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(gamma="scale", random_state=42)

    # 投票分类器，将选择票数最多的类作为最终的分类
    voting_clf = VotingClassifier(
        estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
        voting="hard",
    )

    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        logger.info(
            "model:{} , accuracy:{}".format(
                clf.__class__.__name__, accuracy_score(y_test, y_pred)
            )
        )


def run():
    X_train, X_test, y_train, y_test = get_data()
    fit_predict_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run()
