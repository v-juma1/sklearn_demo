import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectKBest


"""
对糖尿病进行预测
"""

data=pd.read_csv(r"data\pima-indians-diabetes\diabetes.csv")
#数据特征的解释
# Pregnancies  怀孕的次数
# Glucose  血浆葡萄糖浓度
# BloodPressure  舒张压
# SkinThickness  肱三头肌皮肤褶皱厚度
# Insulin   两小时血清胰岛素
# BMI  身体质量指数
# DiabetesPedigreeFunction  糖尿病血统指数
# Age  年龄
# Outcome  标记值，1表示有糖尿病 

x=data.iloc[:,0:8]
y=data.iloc[:,8]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

models=[]
#普通的k均值算法
models.append(("knn",KNeighborsClassifier(n_neighbors=2)))

#带权重的k均值算法
models.append(("knn_weights",KNeighborsClassifier(n_neighbors=2,weights="distance")))

#指定范围的k均值算法
models.append(("knn_radius",RadiusNeighborsClassifier(n_neighbors=2,radius=500.0)))

# #分别训练三个模型并计算评分，比较那哪种模型的效果好
# #多次随机分配训练数据和测试数据，求模型准确性评分的平均值
# result=[]
# for name ,model in models:
#     model.fit(x_train,y_train)
#     kfold=KFold(n_splits=10)
#     #cross_val_score(model,x,y,cv=kfold)得到10次准确性评分的数组
#     cv_result=cross_val_score(model,x,y,cv=kfold).mean()
#     result.append((name,cv_result))

# print(result)

#经过以上比较，发现普通的k均值性能更好
#画出其学习曲线
#训练样本的数量按照规定的量不断的增加，画出不同训练样本数量时模型的准确性
#train_sizes=np.linspace(0.1,1.0,5)表示从0.1到1分成5等份
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

cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
plt.figure(figsize=(10,6),dpi=200)
plot_learning_curve(plt,models[0][1],"learning curve for diabetes",x,y,ylim=(0.0,1.01),cv=cv)
plt.show()

#选出两个与输出值相关性最大的特征，其原理是使用卡方检验，详见https://www.zhihu.com/question/63191726
selector=SelectKBest(k=2)
x_best=selector.fit_transform(x,y)

#用选出来的2个相关性最强的特征训练三种不同的k均值模型
result=[]
for name ,model in models:
    model.fit(x_train,y_train)
    kfold=KFold(n_splits=10)
    #cross_val_score(model,x,y,cv=kfold)得到10次准确性评分的数组
    cv_result=cross_val_score(model,x,y,cv=kfold).mean()
    result.append((name,cv_result))

print(result)