import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

#生成用于k-means分类的数据
#3个数据中心点
centers=[[-2,2],[2,2],[0,4]]

#数据和数据的类别标记,200个点，标准差为0.60
x,y=make_blobs(n_samples=200,centers=centers,random_state=0,cluster_std=0.6)

c=np.array(centers)

#画出数据
#plt.figure(figsize=(16,10),dpi=144)
# #X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，第二维中取第0个数据
# plt.scatter(x[:,0],x[:,1],c=y,s=100,cmap='cool') #样本
# plt.scatter(c[:,0],c[:,1],s=100,marker='^',c='orange')#中心点
# plt.show()

#预定义参数k(距离最近的k个点)，待标记样本的类别，有距离最近的k个样本投票产生
k=5
clf=KNeighborsClassifier(n_neighbors=k)
clf.fit(x,y)

#对一个样本进行预测，并且画出样本周围距离最近的k个点
x_point=[-1,2]
x_point = np.array(x_point).reshape(1, -1)
y_point=clf.predict(x_point)
neighbors=clf.kneighbors(x_point,return_distance=False)

plt.figure(figsize=(16,10),dpi=144)
plt.scatter(x[:,0],x[:,1],c=y,s=100,cmap='cool')#样本
plt.scatter(c[:,0],c[:,1],s=100,marker='^',c='k')#中心点
plt.scatter(x_point[0][0],x_point[0][1],marker='x',s=100,cmap='cool')#待预测的点

for i in neighbors[0]:
    # 预测点与距离最近的 5 个样本的连线
    plt.plot([x[i][0],x_point[0][0]],[x[i][1],x_point[0][1]],'k--',linewidth=0.6)

plt.show()



