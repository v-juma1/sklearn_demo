import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
"""
乳腺癌预测的svm模型
"""

#数据
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

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
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '.--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#画出模型评分随训练数据变化的曲线图
def plot_learning_curve(plt,estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("training sets")
    plt.ylabel("score")

    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    
    #模型训练的样本是随机分配，每次训练的准确率不一样，选择用均值和方差来描述
    train_score_mean=np.mean(train_scores,axis=1)
    train_score_std=np.std(train_scores,axis=1)
    test_score_mean=np.mean(test_scores,axis=1)
    test_score_std=np.std(test_scores,axis=1)

    plt.grid()

    #将均值上下方差的空间用颜色填充
    plt.fill_between(train_sizes,train_score_mean-train_score_std,train_score_mean+train_score_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.1,color="g")

    plt.plot(train_sizes,train_score_mean,'o-',color="r",label="training score")
    plt.plot(train_sizes,test_score_mean,'o-',color="g",label="validation score")

    plt.legend(loc="best")
    return plt

#高斯核SVM
def rbf_model():
    #使用高斯核，用GridSearchCV确定最优的gamma值
    gammas=np.linspace(0,0.0003,30)
    param={"gamma":gammas}
    clf=GridSearchCV(SVC(),param,cv=5,return_train_score=True)
    clf.fit(x,y)

    #画出模型评分随gamma值变化的函数图
    plt.figure(figsize=(10,4),dpi=144)
    plot_param_curve(plt,gammas,clf.cv_results_,xlabel="gamma")
    plt.show()

    #画出gamma值为0.01时模型评分随训练数据变化的曲线图,会看到发生了明显的过拟合
    cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
    title="Gaussian kernel model"
    plt.figure(figsize=(10,4),dpi=144)
    plot_learning_curve(plt,SVC(C=1.0,kernel="rbf",gamma=0.01),title,x,y,ylim=(0.5,1.01),cv=cv)
    plt.show()

#多项式核SVM
def poly_model():
    #画出一阶多项式核、二阶多项式核的拟合情况
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    title="degree={0}"
    degrees=[1,2]

    plt.figure(figsize=(12,4),dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1,len(degrees),i+1)
        plot_learning_curve(plt,SVC(C=1.0,kernel="poly",degree=degrees[i]),title.format(degrees[i]),x,y,ylim=(0.8,1.01), cv=cv,n_jobs=4)
    
    plt.show()

poly_model()







