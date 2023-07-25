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
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_blobs
"""
通过生成的数据了解SVM基本参数
"""


#生成数据100个点，分三类
def generate_data(random_state=0):

    x, y = make_blobs(n_samples=100,
                      centers=3,
                      random_state=random_state,
                      cluster_std=0.8)
    return x, y


#画出支持向量
def plot_hyperplane(clf, X, y, h=0.02, draw_sv=True, title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y == label][:, 0],
                    X[y == label][:, 1],
                    c=colors[label],
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


#选择不同的核来构建四个不同的模型
def get_4models():
    clf_linear = svm.SVC(C=1.0, kernel="linear")
    clf_poly = svm.SVC(C=1.0, kernel="poly", degree=3)
    clf_rbf = svm.SVC(C=1.0, kernel="rbf", gamma=0.5)
    clf_rbf1 = svm.SVC(C=1.0, kernel="rbf", gamma=0.1)

    clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf1]
    return clfs


#画出四个模型的分割超平面和支持向量
def plt_4models(models, x, y):
    titles = [
        'Linear Kernel', 'Polynomial Kernel with Degree=3',
        'Gaussian Kernel with $\gamma=0.5$',
        'Gaussian Kernel with $\gamma=0.1$'
    ]
    plt.figure(figsize=(10, 10), dpi=144)
    for clf, i in zip(models, range(len(models))):
        clf.fit(x, y)
        plt.subplot(2, 2, i + 1)
        plot_hyperplane(clf, x, y, title=titles[i])
    plt.savefig("scikit-learn/log/4-model.png")


#画出模型评分随某些参数变化的曲线图
def plot_param_curve(plt, train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")
    plt.plot(train_sizes,
             train_scores_mean,
             '.--',
             color="r",
             label="Training score")
    plt.plot(train_sizes,
             test_scores_mean,
             '.-',
             color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#画出模型评分随训练数据变化的曲线图
def plot_learning_curve(plt,
                        estimator,
                        title,
                        x,
                        y,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("training sets")
    plt.ylabel("score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    #模型训练的样本是随机分配，每次训练的准确率不一样，选择用均值和方差来描述
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)

    plt.grid()

    #将均值上下方差的空间用颜色填充
    plt.fill_between(train_sizes,
                     train_score_mean - train_score_std,
                     train_score_mean + train_score_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_score_mean - test_score_std,
                     test_score_mean + test_score_std,
                     alpha=0.1,
                     color="g")

    plt.plot(train_sizes,
             train_score_mean,
             'o-',
             color="r",
             label="training score")
    plt.plot(train_sizes,
             test_score_mean,
             'o-',
             color="g",
             label="validation score")

    plt.legend(loc="best")
    return plt


#高斯核SVM
def rbf_model(x, y):
    #使用高斯核，用GridSearchCV确定最优的gamma值
    gammas = np.linspace(0, 0.0003, 30)
    param = {"gamma": gammas}
    clf = GridSearchCV(SVC(), param, cv=5, return_train_score=True)
    clf.fit(x, y)

    #画出模型评分随gamma值变化的函数图
    plt.figure(figsize=(10, 4), dpi=144)
    plot_param_curve(plt, gammas, clf.cv_results_, xlabel="gamma")
    plt.savefig("scikit-learn/log/rbf_model.png")

    #画出gamma值为0.01时模型评分随训练数据变化的曲线图,会看到发生了明显的过拟合
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "Gaussian kernel model"
    plt.figure(figsize=(10, 4), dpi=144)
    plot_learning_curve(plt,
                        SVC(C=1.0, kernel="rbf", gamma=0.01),
                        title,
                        x,
                        y,
                        ylim=(0.5, 1.01),
                        cv=cv)
    plt.savefig("scikit-learn/log/rbf_model_gamma_0.01.png")


#多项式核SVM
def poly_model(x, y):
    #画出一阶多项式核、二阶多项式核的拟合情况
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    title = "degree={0}"
    degrees = [1, 2]

    plt.figure(figsize=(12, 4), dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1, len(degrees), i + 1)
        plot_learning_curve(plt,
                            SVC(C=1.0, kernel="poly", degree=degrees[i]),
                            title.format(degrees[i]),
                            x,
                            y,
                            ylim=(0.8, 1.01),
                            cv=cv,
                            n_jobs=4)

    plt.savefig("scikit-learn/log/poly.png")


def run():
    x, y = generate_data()
    models = get_4models()
    plt_4models(models, x, y)
    poly_model(x, y)
    rbf_model(x, y)


if __name__ == "__main__":
    run()
