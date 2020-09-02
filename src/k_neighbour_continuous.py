import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

"""
使用k_邻近对连续区间内的数值进行回归拟合
"""

#生成数据
n_dots=200
x=5*np.random.rand(n_dots,1)
y=np.cos(x).ravel()

#噪声点
y+=0.2*np.random.rand(n_dots)-0.1

k=5
knn=KNeighborsRegressor(k)
knn.fit(x,y)

#在x轴上指定区间生成足够多的点，使用模型预测这些点的y值，然后将所有预测点连接起来形成曲线
t=np.linspace(0,5,1000)[:,np.newaxis]
y_pred=knn.predict(t)

plt.figure(figsize=(16,10),dpi=144)
plt.scatter(x,y,c='g',label="data",s=100)
plt.plot(t,y_pred,c='k',label="prediccion",lw=4)
plt.axis("tight")
plt.title("KNeighborsRegressor")
plt.show()