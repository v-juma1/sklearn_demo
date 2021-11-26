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
import matplotlib.pyplot as plt
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

digits = datasets.load_digits()

# #show minist data
# data=list(zip(digits.images,digits.target))
# plt.figure(figsize=(8,6),dpi=200)
# for index,(image,label) in enumerate(data[:8]):
#     plt.subplot(2,4,index+1)
#     plt.axis('off')
#     plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
#     plt.title('digits:%i' % label,fontsize=20)
# plt.show()

#将数据中的20%作为测试集，剩下的作为训练集
xtrain, xtest, ytrain, ytest = train_test_split(digits.data,
                                                digits.target,
                                                test_size=0.20,
                                                random_state=2)

#使用支持向量机分类
# C=1.0 对误分类的惩罚参数，惩罚参数越大，模型的准确率越高，但泛化能力越弱，反之，相当于把错误分类的样本看成噪声点
# kernel='rbf' 核函数，用来计算映射到高维空间之后的内积（相似度）的一种简便方法
#   ‘linear’:线性核函数
#   ‘poly’：多项式核函数
#   ‘rbf’：径像核函数/高斯核
#   ‘sigmod’:sigmod核函数
#   ‘precomputed’:核矩阵
# degree=3 多项式核函数poly函数的维度，默认是3，选择其他核函数时会被忽略
# gamma='scale'rbf,poly,sigmod核函数的参数
#   如果为auto，代表其值为样本特征数的倒数，即1/n_features
#   如果为scale，则是 1/(n_features * X.var())
# coef0=0.0 核函数中的独立项，只有对‘poly’和‘sigmod’核函数有用，是指其中的参数c
# shrinking=True 是否采用shrinking heuristic（启发收缩）方法
# probability=False 是否采用概率估计，会使fit方法变慢
# tol=1e-3 SVM停止训练的误差精度
# cache_size=200 训练所需内存,MB为单位
# class_weight=None, 对不同类设置不同权重的惩罚参数
#   如果给定参数‘balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重
# verbose=False 是否启用详细输出
# max_iter=-1 最大迭代次数，-1表示不限制
# decision_function_shape='ovr' 取值ovo’, ‘ovr’ or None
# break_ties=False,
# random_state=None

clf = svm.SVC(C=1.0, gamma=0.001, kernel='rbf')
clf.fit(xtrain, ytrain)

# #正确率
# print(clf.score(xtest,ytest))

#保存模型
savedir = r"scikit-learn/model/digits_svm.pkl"
joblib.dump(clf, savedir)

#导入模型，进行预测
clf = joblib.load(savedir)
ypred = clf.predict(xtest)
logger.info(clf.score(xtest, ypred))
