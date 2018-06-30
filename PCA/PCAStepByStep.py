import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100,2))  # 100个样本，2个特征
np.random.seed(1)
X[:,0] = np.random.uniform(0., 100, size=100)
X[:,1] = 0.75 * X[:,0] +3. + np.random.normal(0,10,size=100)

#样本均值归零
def demean(X):
    return X - np.mean(X, axis=0)# 每一列(特征)的均值

#样本X映射到w向量方向后，其方差
def f(w, X):
    return np.sum((X.dot(w) ** 2)) / len(X)

#对f求梯度上升，求其最大值
def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

#单位化向量w
def direction(w):
    return w / np.linalg.norm(w)

def gradient_ascent(df, X, init_w, eta, max_iter=1e4, epsilon=1e-8):
    w = direction(init_w)
    curr_iter = 1
    while curr_iter < max_iter:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient   #梯度上升法
        w = direction(w) #注意1：单位化向量w,表示单位方向
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        curr_iter += 1
    return w

np.random.seed(1)
init_w = np.random.random(X.shape[1])  #注意2：初始化init_w，不能是全0向量，没法求导
eta = 0.001

#注意3：不能使用StandardScaler标准化数据

#样本均值归零
X1 = demean(X)

w1 = gradient_ascent(df, X1, init_w, eta)
print("w1:", w1)
# w1: [0.79462549 0.60709994]
plt.scatter(X1[:,0], X1[:,1])
plt.plot([0, w1[0]*30], [0, w1[1]*30], color='r')
plt.show()

# X1去除w1方向分量：X2
X2 = X1 - X1.dot(w1).reshape(-1, 1) * w1

w2 = gradient_ascent(df, X2,init_w, eta)
print("w2:", w2)
# w2: [-0.6070763   0.79464354]
plt.scatter(X2[:,0], X2[:,1])
plt.plot([0, w2[0]*10], [0, w2[1]*10], color='r')
plt.show()

# w1和w2是正交的，点乘=cos90=0
print("w1.dot(w2):", w1.dot(w2))
# w1.dot(w2): 2.9746216804460435e-05

#########################################
#调用自己写的PCA类尝试
from PCA.PCAClassfier import PCAClassfier
pca = PCAClassfier(n_components=2)
pca.fit(X)
print("n_components=2,pca.components_:", pca.components_)
#n_components=2,pca.components_: [[ 0.7946258   0.60709953] [ 0.60710555 -0.7946212 ]]


#2个特征，PCA降维到1个特征
pca = PCAClassfier(n_components=1)
pca.fit(X)
print("self pca.components_:",pca.components_)
#self pca.components_: [[0.79462588 0.60709942]]

X_reduction = pca.transform(X)

#基于降维的特征，inverse恢复与原X做对比，看看损失多少信息
X_restore = pca.inverse_transform(X_reduction)

plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()

#########################################
#scikit-learn的PCA类尝试
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X)
#scikit learn中PCA并不是使用梯度上升法，而是用的方程求解法，其结果与自己写的相反，但不影响
print("scikit learn pca.components_:",pca.components_)
#scikit learn pca.components_: [[-0.79462592 -0.60709937]]

X_reduction = pca.transform(X)

#基于降维的特征，inverse恢复与原X做对比，看看损失多少信息
X_restore = pca.inverse_transform(X_reduction)

plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()




