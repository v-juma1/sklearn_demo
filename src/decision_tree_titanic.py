import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

"""
用决策树来预测泰坦尼克号的幸存者
"""

#读取数据集
def read_dataset(fname):
    # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0) 
    # 丢弃无用的数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 处理性别数据
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # 处理登船港口数据
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)
    return data

train = read_dataset('data/titanic/train.csv')

y= train["Survived"].values
x=train.drop(["Survived"],axis=1).values

#分割数据x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#DecisionTreeClassifier()模型常见参数
# criterion="gini", 特征选择算法，gini(基尼不纯度)或者entropy(信息熵),两者差异不大
# splitter="best",  best ：选择最优的分支创建，random：在排名靠前的特征种随机选择一个来创建
# max_depth=None, 决策树的最大深度
# min_samples_split=2, 能创建分支的数据集大小，一个节点的数据样本小于这个数，将不再创建分支
# min_samples_leaf=1, 创建分支后，节点的样本量必须大于或者等于这个值，否则将不再创建分支
# min_weight_fraction_leaf=0.,限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝
# max_leaf_nodes=None, 最大的样本节点个数
# min_impurity_split=None, 即将废弃,信息增益（划分数据集前后信息熵的变化值）的阈值，小于这个阈值将不创建分支
# min_impurity_decrease,代替min_impurity_split作为信息增益的参数
# class_weight=None, 指定各类样本的权重


#模型的优化方法，选择某个参数，在一组范围确定这个参数的最优值
#比如max_depth（模型深度），min_impurity_split（信息增益的阈值），确定这两个参数的最优值，并且画出其和模型评分的关系

#找出得分最高的max_depth，并画出其和模型评分的关系
def max_depth_score():
    def cv_score(d):
        clf=DecisionTreeClassifier(max_depth=d)
        clf.fit(x_train,y_train)
        train_score= clf.score(x_train,y_train)
        test_score=clf.score(x_test,y_test)
        return (train_score,test_score)
    
    depths=range(2,15)
    scores=[cv_score(d) for d in depths]
    train_scores=[s[0] for s in scores]
    test_scores=[s[1] for s in scores]

    #找出测试集上score最高的max_depth
    best_score_index=np.argmax(test_scores)
    best_score=test_scores[best_score_index]
    best_param=depths[best_score_index]
    #print("depth:{},score:{}".format(best_param,best_score)

    #画出关系图
    plt.figure(figsize=(6,4),dpi=144)
    plt.grid()
    plt.xlabel("max depth of decision tree")
    plt.ylabel("score")
    plt.plot(depths,train_scores,'.g-',label='train score')
    plt.plot(depths,test_scores,'.r--',label="test score")
    plt.legend()
    plt.show()

#找出得分最高的min_impurity_split，并画出其和模型评分的关系
def min_impurity_split_score():
    def cv_score(val):
        clf=DecisionTreeClassifier(criterion="gini",min_impurity_split=val)
        clf.fit(x_train,y_train)
        train_score= clf.score(x_train,y_train)
        test_score=clf.score(x_test,y_test)
        return (train_score,test_score)
    
    values=np.linspace(0,0.5,50)
    scores=[cv_score(d) for d in values]
    train_scores=[s[0] for s in scores]
    test_scores=[s[1] for s in scores]

    #找出测试集上score最高的max_depth
    best_score_index=np.argmax(test_scores)
    best_score=test_scores[best_score_index]
    best_param=values[best_score_index]
    #print("min_impurity_split:{},score:{}".format(best_param,best_score)

    #画出关系图
    plt.figure(figsize=(6,4),dpi=144)
    plt.grid()
    plt.xlabel("min_impurity_split of decision tree")
    plt.ylabel("score")
    plt.plot(values,train_scores,'.g-',label='train score')
    plt.plot(values,test_scores,'.r--',label="test score")
    plt.legend()
    plt.show()

#以上挨个寻找最优参数的方法有两个问题
# 1. 数据不稳定
# 2. 不能一次选择多个参数最优的参数组合
# 使用sklean的模型参数选择工具包GridSearchCV

def GridSearchCV_curve():
    #画出模型参数于模型评分的关系图
    def plot_curve(train_size,cv_result,xlabel):
        mean_train=cv_result["mean_train_score"]
        std_train=cv_result["std_train_score"]
        mean_test=cv_result["mean_test_score"]
        std_test=cv_result["std_test_score"]



        plt.figure(figsize=(6,4),dpi=144)
        plt.title("parameters turning")
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel("score")

        #填充上下两个方差的区间
        plt.fill_between(train_size,mean_train-std_train,mean_train+std_train,alpha=0.1,color='r')
        
        plt.fill_between(train_size,mean_test-std_test,mean_test+std_test,alpha=0.1,color='g')

        plt.plot(train_size,cv_result["mean_train_score"],'.--',color='r',label="train score")
        plt.plot(train_size,cv_result["mean_test_score"],'.-',color='g',label="test score")
        plt.legend(loc="best")
        plt.show()

    space=np.linspace(0,0.5,50)
    
    #多个参数同时调优，可以用列表的形式：
    # params=[{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},{'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
    #   {'max_depth': range(2, 10)},{'min_samples_split': range(2, 30, 2)}]
    params={'min_impurity_split':space}

    #GridSearchCV会根据params中的值来构建模型，其中并计算平均分和标准差，cv=5表示每次计算把数据分为5份，一份测试，其他用来训练
    #得到的最优参数及评分保存在clf.best_params_和clf.best_score_中，clf.cv_results_保存了计算过程中的中间结果
    clf=GridSearchCV(DecisionTreeClassifier(),params,cv=5,return_train_score=True)
    clf.fit(x,y)
    print("params:{0}\n score:{1}".format(clf.best_params_,clf.best_score_))

    plot_curve(space, clf.cv_results_, xlabel='gini thresholds')



GridSearchCV_curve()