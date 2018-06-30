import numpy as np

class PCAClassfier:

    def __init__(self, n_components):
        assert n_components >= 1, "待降维的数据维度需要>= 1 ！"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, max_iter=1e4):
        assert self.n_components <= X.shape[1], "待降维的数据维度不能大于X的feature个数！"

        #样本均值归零函数
        def demean(X):
            return X - np.mean(X, axis=0)# 每一列(特征)的均值。每个样本都以此为准将其所有特征均值归零

        #样本X映射到w向量方向后，其方差函数
        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        #对f求梯度上升函数
        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        #单位化向量w
        def direction(w):
            return w / np.linalg.norm(w)

        #梯度上升法，求f的最大值函数
        def gradient_ascent(X, init_w, eta=0.01, max_iter=1e4, epsilon=1e-8):
            #单位化向量w
            w = direction(init_w)
            curr_iter = 1
            while curr_iter < max_iter:
                #对f求梯度上升，求其最大值
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient   #梯度上升法
                w = direction(w) #注意1：单位化向量w,表示单位方向
                if(abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                curr_iter += 1
            return w

        #第一步：样本均值归零
        X_pca = demean(X)
        #最终求解的PCA结果self.components_ 初始化  有n_components行，X的特征数量的列
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            #注意：初始化init_w，不能是全0向量，没法求导
            init_w = np.random.random(X_pca.shape[1])
            #调用梯度上升法，求第i个特征映射的分量方向w
            w = gradient_ascent(X_pca, init_w, eta, max_iter)
            #计算出的第i个PCA存入components_
            self.components_[i,:] = w
            # X再去除第i个PCA方向分量后形成的新X_pca，继续在下一个分量方向求PCA
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    #将X进行PCA，得到降维后的矩阵
    def transform(self, X):
        assert X.shape[1] == self.components_.shape[1], "PCA的列数必须与X的特征feature数量相等！"
        return X.dot(self.components_.T)

    # 降维后的矩阵X，逆PCA，反向映射回来原来的特征空间，看与原矩阵的变化大小
    def inverse_transform(self, X):
        assert X.shape[1] == self.components_.shape[0], "PCA的列数必须与待逆PCA矩阵的特征feature数量相等！"
        return X.dot(self.components_)


    def __repr__(self):
        return "PCA(n_components=%d" % self.n_components
