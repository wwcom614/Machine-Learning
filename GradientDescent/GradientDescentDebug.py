import numpy as np
from Utils.AccuracyFunction import r2_score

class GradientDescentDebug:
    def __init__(self):
        self.coefficient_  = None  #系数
        self.interception_ = None  #截距
        self._theta = None

    #使用批量梯度下降法求解最小损失函数，比常规方程解好。eta是每次求解移动的步长，max_iter是梯度下降最大迭代次数
    #但批量梯度下降法随样本数m增大，计算量还是增大的，所以考虑“随机梯度下降法”，进一步降低样本数m增大对计算法增大的影响
    def fit_gd(self, X_train, y_train, eta = 0.01, max_iter = 1e4):
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be the same as y_train !"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.zeros(X_b.shape[1])

        #theta是待求解的多元线性回归的参数
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ_debug(theta, X_b, y, epsilon=0.01):
            #在每个theta点两边极近相等距离取2个点，这两个点直线斜率约等于其导数。
            #用途：该方法是通用的，适用于所有函数求梯度，不限于线性回归。但性能不好，可用于测试验证看自己求导的结果于此是否一致
            res = np.empty(len(theta))
            for i in range(len(theta)):
                theta_1 = theta.copy()
                theta_1 += epsilon
                theta_2 = theta.copy()
                theta_2 += epsilon
                res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y) / (2 * epsilon))
            return res

        def gradient_descent(X_b, y, init_theta, eta, max_iter=1e4, epsilon=1e-6):
            theta = init_theta
            curr_iter = 1

            while curr_iter < max_iter:
                gradient = dJ_debug(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                curr_iter += 1
            return theta

        self._theta = gradient_descent(X_b, y_train, init_theta, eta)
        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]

        return self



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