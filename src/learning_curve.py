import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


#画出模型（y=√x）学习曲线，从而判断模型的优化方向
#准确率作为纵坐标，训练集个数作为横坐标

#生成数据
n_dots=200
x=np.linspace(0,1,n_dots)
y=np.sqrt(x)+0.2*np.random.rand(n_dots)-0.1

x=x.reshape(-1,1)
y=y.reshape(-1,1)

def polynomial_model(degree=1):
    #根据给定的特征，进行特征的多项式组合
    #degree表示多项式的阶数，即多项式中最高次项的阶数
    #例如，输入特征[a,b]，degree=2,则组合特征为[1, a, b, a^2, ab, b^2]
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)

    #可以通过normalize参数指定是否对数据进行归一化
    linear_regression=LinearRegression()

    pipeline=Pipeline([("polynomial_features",polynomial_features),("linear_regression",linear_regression)])
    return pipeline

#训练样本的数量按照规定的量不断的增加，画出不同训练样本数量时模型的准确性
#train_sizes=np.linspace(0.1,1.0,5)表示从0.1到1分成5等份
def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1.0,5)):
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

#构造出一阶多项式，三阶多项式，十阶多项式三个模型的学习曲线
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
titles=['learning curves_1degree(under fitting)','learning curves_3degree','learning curves_10degree(over fitting)']
degrees=[1,3,10]
plt.figure(figsize=(18,4),dpi=200)

for i in range(len(degrees)):
    plt.subplot(1,3,i+1)
    plot_learning_curve(polynomial_model(degrees[i]),titles[i],x,y,ylim=(0.75,1.01),cv=cv)

plt.show()

