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

import joblib
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import svm


#加载数据
def load_data():
    digits = datasets.load_digits()
    return digits


#查看数据
def show_data(digits):
    data = list(zip(digits.images, digits.target))
    plt.figure(figsize=(8, 6), dpi=200)
    for index, (image, label) in enumerate(data[:8]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('num:%i' % label, fontsize=20)
    plt.savefig("scikit-learn/log/minist_demo.png")
    plt.show()


#将数据中的20%作为测试集，剩下的作为训练集
def split_data(digits, test_size=0.20, random_state=42):
    xtrain, xtest, ytrain, ytest = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=test_size,
                                                    random_state=random_state)
    return xtrain, xtest, ytrain, ytest


#用GridSearchCV寻找最佳参数
#用cross_val_score查看模型性能
def fine_tuning(model, param, datax, datay):
    # 使k折中，每折样本类别的比例大致相同
    sk = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(model, param, cv=sk)
    grid_search.fit(datax, datay)

    # 最佳参数
    logger.info("最佳参数:{%s}", grid_search.best_params_)

    #k折交叉验证
    val_score = cross_val_score(grid_search.best_estimator_,
                                datax,
                                datay,
                                cv=sk,
                                scoring="accuracy")
    logger.info("最佳参数模型平均accuracy:{%s}", np.mean(val_score))
    return grid_search.best_params_


#模型定义
class minist_model:

    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, model, xtrain, ytrain):
        model.fit(xtrain, ytrain)

        #在train数据上的正确率
        logger.info(model.score(xtrain, ytrain))

        # 或者查看模型的混淆矩阵
        # cross_val_predict执行k折交叉验证，但并不是和cross_val_score一样返回每折的分数
        # 而是返回每折的预测
        y_pred = cross_val_predict(
            model,
            xtrain,
            ytrain,
            cv=10,
        )

        #每行代表实际类，每列代表预测类
        matrix = confusion_matrix(y_pred, ytrain)
        logger.info(matrix)

        #将混淆矩阵用图像表示
        plt.matshow(matrix, cmap=plt.cm.cool)
        plt.savefig("scikit-learn/log/row_matrix.png")

        #将混淆举证中的每个值除以相应类别的数量
        #从而凸显错误率而不是错误数量（否则对图片数量较多的类不公平）
        rows_sum = matrix.sum(axis=1, keepdims=True)
        norm_matrix = matrix / rows_sum

        # 用0填充对角线，只保留错误，重新绘图
        np.fill_diagonal(norm_matrix, 0)
        plt.matshow(norm_matrix, cmap=plt.cm.cool)
        plt.savefig("scikit-learn/log/norms_matrix.png")
        #保存模型
        joblib.dump(model, self.model_path)

    def test(self, xtest, ytest):
        #导入模型，进行测试
        clf = joblib.load(self.model_path)
        logger.info(clf.score(xtest, ytest))
        ypred = clf.predict(xtest)

        logger.info(precision_score(ytest, ypred, average="macro"))
        logger.info(recall_score(ytest, ypred, average="macro"))
        logger.info(f1_score(ytest, ypred, average="macro"))

    def predict(self, xtest):
        clf = joblib.load(self.model_path)
        ypred = clf.predict(xtest)
        logger.info(ypred)
        return ypred


def run(model_name, task):

    digits = load_data()
    #show_data(digits)
    xtrain, xtest, ytrain, ytest = split_data(digits)

    #定义模型，待调优的参数，和保存模型文件的路径
    if model_name == "svm":
        #使用支持向量机分类
        # C=1.0 对误分类的惩罚参数，惩罚参数越大，模型的准确率越高，但泛化能力越弱，反之，相当于把错误分类的样本看成噪声点
        # kernel='rbf' 核函数，用来计算映射到高维空间之后的内积（相似度）的一种简便方法
        #   ‘linear’:线性核函数
        #   ‘poly’：多项式核函数
        #   ‘rbf’：径像核函数/高斯核
        #   ‘sigmod’:sigmod核函数
        #   ‘precomputed’:核矩阵
        # gamma='scale'rbf,poly,sigmod核函数的参数
        #   如果为auto，代表其值为样本特征数的倒数，即1/n_features
        #   如果为scale，则是 1/(n_features * X.var())
        pre_model = svm.SVC()
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': ["scale", "auto"]
        }

        model_path = r"scikit-learn/model/digits_svm.pkl"

    elif model_name == "sgd":

        pre_model = SGDClassifier()
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01]
        }

        model_path = r"scikit-learn/model/digits_SGD.pkl"

    model = minist_model(model_path)

    if task == "train":
        #训练之前先找到最优参数，用最优参数初始化
        best_params = fine_tuning(pre_model, param_grid, xtrain, ytrain)
        pre_model = pre_model.set_params(**best_params)
        model.train(pre_model, xtrain, ytrain)

    elif task == "test":
        model.test(xtest, ytest)

    elif task == "predict":
        model.predict(xtest[0:10])


if __name__ == '__main__':
    #run("sgd", "train")
    #run("sgd", "test")
    #run("sgd", "predict")

    run("svm", "train")
    #run("svm", "test")
    #run("svm", "predict")
