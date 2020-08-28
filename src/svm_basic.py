import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
通过生成的数据了解SVM基本参数
"""

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
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


#生成数据100个点，分三类
x,y=make_blobs(n_samples=100,centers=3,random_state=0,cluster_std=0.8)

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

#选择不同的核来构建四个不同的模型
clf_linear=svm.SVC(C=1.0,kernel="linear")
clf_poly=svm.SVC(C=1.0,kernel="poly",degree=3)
clf_rbf=svm.SVC(C=1.0,kernel="rbf",gamma=0.5)
clf_rbf1=svm.SVC(C=1.0,kernel="rbf",gamma=0.1)

clfs=[clf_linear,clf_poly,clf_rbf,clf_rbf1]

#画出四个模型的分割超平面和支持向量
titles=['Linear Kernel', 
          'Polynomial Kernel with Degree=3', 
          'Gaussian Kernel with $\gamma=0.5$', 
          'Gaussian Kernel with $\gamma=0.1$']
plt.figure(figsize=(10,10),dpi=144)
for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(x, y)
    plt.subplot(2, 2, i+1)
    plot_hyperplane(clf, x, y, title=titles[i])
plt.show()
