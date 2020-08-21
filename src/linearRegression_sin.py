import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams

"""
使用线性回归模拟正弦函数
"""

#生成数据
n_dots=200
x=np.linspace(-2*np.pi,2*np.pi,n_dots)
y=np.sin(x)+0.2*np.random.rand(n_dots)-0.1
x=x.reshape(-1,1)
y=y.reshape(-1,1)

def polynomial_model(degree=1):
    pf=PolynomialFeatures(degree=degree,include_bias=False)
    lr=LinearRegression(normalize=True)
    pipe=Pipeline([("polynomial_feature",pf),("linear_regression",lr)])
    return pipe

#2,3,5,10阶的多项式特征
degrees=[2,3,5,10]
results=[]
for d in degrees:
    model=polynomial_model(d)
    model.fit(x,y)
    train_score=model.score(x,y)
    #均方根误差，实际的点和模型预测的点之间的距离,值越小，模型越好
    mse=mean_squared_error(y,model.predict(x))
    results.append({"model":model,"degree":d,"mse":mse})

#print(results)

#画出四个模型的拟合效果
plt.figure(figsize=(12,6),dpi=200,subplotpars=SubplotParams(hspace=0.3))
for i,r in enumerate(results):
    fig=plt.subplot(2,2,i+1)
    plt.xlim(-8,8)
    plt.title("degree="+str(r["degree"]))
    plt.scatter(x,y,s=5,c='b',alpha=0.5)
    plt.plot(x,r["model"].predict(x),"r--")


plt.show()
