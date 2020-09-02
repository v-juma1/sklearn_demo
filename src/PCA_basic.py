import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

"""
使用PCA对数据进行降维
"""


a=np.array([[3,2000],[2,3000],[4,5000],[5,8000],[1,2000]],dtype='float')

#使用numpy自定义pca函数
def pca_fun(a):
    #数据中两个维度的值不在同一个数量级，采用归一化和缩放提高PCA的效率
    #归一化：每一维的特征值减去该维特征的均值，使得改维特征最后的均值为0
    #缩放：在以上的基础上，每一维特征除以该维特征的范围(范围：max-min)

    #归一化
    mean=np.mean(a,axis=0)
    norm=a-mean

    #缩放
    scope=np.max(norm,axis=0)-np.min(norm,axis=0)
    norm=norm/scope
    pca_fun(a)
    #对norm的协方差矩阵进行SVD奇异值分解
    #np.dot(norm.T,norm)即 a的转置xa 可以将a从矩阵转化为方阵
    #u中的列向量即为协方差矩阵的特征向量
    u,s,v=np.linalg.svd(np.dot(norm.T,norm))

    #从u中选取k列（本例中选一列）构成主成分特征矩阵
    u_reduce=u[:,0].reshape(2,1)

    #使用主成分特征矩阵，对数据进行降维(本例中从二维将到一维)
    r=np.dot(norm,u_reduce)

    #对降维的数据进行还原
    z=np.dot(r,u_reduce.T)
    origin=np.multiply(z,scope)+mean

#使用sklearn进行pca降维
def sk_pca(a):
    scaler=MinMaxScaler()

    # n_components=None, 指定降维后的特征维度
    #    最常用的做法是直接指定降维到的维度数目，个大于等于1的整数
    #    也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数
    #    还可以将参数设置为'mle'(极大似然估计)， 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维
    #    也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)
    # copy=True, 在副本上进行运算，不改变原数据
    # whiten=False, 对每个特征进行标准化，让方差都为1
    # svd_solver='auto', svd的方法，有四个参数可选
    #    'randomized' 一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。
    #    'full' 则是传统意义上的SVD，使用了scipy库对应的实现。
    #    'arpack' 和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现
    #    当svd_solve设置为'arpack'时，保留的成分必须少于特征数，即不能保留所有成分。
    #    'auto'，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。
    
    # 注意：当设置 n_components == 'mle'时，需要和参数svd_solver一起使用，且svd_solver需要选择 'full' 参数；即pca = PCA(n_components = 'mle',svd_solver='full')
    #    同时要保证输入数据的样本数多于特征数才可执行成功。
    
    #    有两个PCA类的成员值得关注。
    #    第一个是explained_variance_，它代表降维后的各主成分的方差值，方差值越大，则说明越是重要的主成分
    #    第二个是explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分

    p=PCA(n_components=1)
    pipe=Pipeline([("scaler",scaler),("pca",p)])
    
    #降维结果
    r=pipe.fit_transform(a)

    #数据恢复
    origin=p.inverse_transform(r)


    print(origin)

sk_pca(a)





