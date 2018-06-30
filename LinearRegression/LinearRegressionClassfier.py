import numpy as np
from Utils.AccuracyFunction import r2_score

class LinearRegressionClassfier:
    def __init__(self):
        self.coefficient_  = None  #系数
        self.interception_ = None  #截距
        self._theta = None

    # 最小损失函数的正规方程解。缺点：时间复杂度高 O(n的3次方，算法优化后也有n的2.4次方)，不推荐使用，推荐使用梯度下降法
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be the same as y_train !"
        #正规方程解，计算量大(n的三次方)，建议使用梯度下降法
        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]
        return self


    #正规方程解通过矩阵乘法和求逆运算来计算参数。当变量很多的时候计算量会非常大
    #使用批量梯度下降法求解最小损失函数，比常规方程解好。
    #eta是每次求解移动的步长，max_iter是梯度下降最大迭代次数
    #但批量梯度下降法随样本数m增大，计算量还是增大的，所以考虑“随机梯度下降法”，进一步降低样本数m增大对计算法增大的影响
    def fit_gd(self, X_train, y_train, eta = 0.01, max_iter = 1e4):
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be the same as y_train !"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.zeros(X_b.shape[1])

        #损失函数J。其中，theta是待求解的多元线性回归的参数
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        #损失函数J求导(极值)
        def dJ(theta, X_b, y):
            ''' 非向量化方式
            res = np.empty(len(theta))
            res[0] = sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
            return res * 2 / len(X_b)
            '''
            #向量化方式
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def gradient_descent(X_b, y, init_theta, eta, max_iter=1e4, epsilon=1e-6):
            theta = init_theta
            curr_iter = 1

            while curr_iter < max_iter:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                # 求导，极小值方向
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                curr_iter += 1
            return theta

        self._theta = gradient_descent(X_b, y_train, init_theta, eta)
        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]

        return self

    #随机梯度下降法。max_iter的含义变为需要对样本遍历几遍，例如样本数为m，max_iter = 5的含义是，遍历m*5次
    def fit_sgd(self, X_train, y_train, max_iter = 5, t0=5, t1=50):
        assert X_train.shape[0] ==y_train.shape[0], "The size of X_train must be the same as y_train !"
        assert max_iter >= 1,"Suggest loop the size of samples at least once !"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.random.randn(X_b.shape[1])

        #相对梯度下降法，随机梯度下降法不用矩阵相乘了，而是矩阵中随机抽取第i行相乘，所以最后也不用除以样本数量m了
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        #批量梯度下降法每次迭代都用所有样本，快速收敛，稳定，但性能不高，
        #随机梯度下降法每次用一个样本调整参数，降低计算量，计算量不再受样本数量影响，逐渐逼近，效率高,随机还有可能跳出局部最优解
        #但每次减小效果的不像批量梯度下降法那么好(稳定)，但经过验证还是能找到最小损失函数的
        #如果步长固定，随机梯度下降法最后可能会在最小损失函数附近来回波动却找不到，所以步长需要开始大，越向后越小
        #最简单的想法是步长eta = t0/(当前迭代次数 + t1)--模拟退火思想
        #a和b是随机梯度下降法的超参数，为避免算法最开始步长下降太大，经验值t0=5,t1=50
        #随机梯度下降法也不用判断两次损失函数之间的最小差值epsilon了--因为不是绝对梯度，不能保证一直下降
        def sgd(X_b, y, init_theta, max_iter=1e4, t0=5, t1=50):
            theta = init_theta
            m = len(X_b) #样本数量

            def learning_rate(x):
                return t0 / (t1 + 50)

            for curr_iter in range(max_iter):
                #既要随机样本，又要遍历所有样本的方案：索引随机生成新数组，顺序遍历该新数组
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    #不是对整个矩阵求梯度，而是X_b随机找一行向量求导
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(curr_iter * m + i) * gradient

            return theta

        self._theta = sgd(X_b, y_train, init_theta, max_iter, t0, t1)
        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]



    def predict(self,X_predict):
        assert self.interception_ is not None and self.coefficient_ is not None, "Before predict must fit ！"
        assert X_predict.shape[1] == len(self.coefficient_), "The features of X_predict and coefficient_ must be the same !"
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Linear Regression"