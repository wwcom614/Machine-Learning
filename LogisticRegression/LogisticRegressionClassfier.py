import numpy as np
from Utils.AccuracyFunction import accuracy_score

class LogisticRegressionClassfier:
    def __init__(self):
        self.coefficient_  = None  #系数
        self.interception_ = None  #截距
        self._theta = None

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    #使用批量梯度下降法求解最小损失函数。
    #eta是每次求解移动的步长，max_iter是梯度下降最大迭代次数
    def fit(self, X_train, y_train, eta = 0.01, max_iter = 1e4):
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be the same as y_train !"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.zeros(X_b.shape[1])

        #损失函数J。其中，theta是待求解的逻辑回归的参数
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta)) # 线性基础上加上sigmoid函数，将预测值域限制在0~1

            try:
                return - np.sum((y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))) / len(y)
            except:
                return float('inf')

        #损失函数J求导(极值)
        def dJ(theta, X_b, y):
            #向量化方式
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, init_theta, eta, max_iter=1e4, epsilon=1e-6):
            theta = init_theta
            curr_iter = 1

            while curr_iter < max_iter:
                gradient = dJ(theta, X_b, y)
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

    def predict_probability(self,X_predict):
        assert self.interception_ is not None and self.coefficient_ is not None, "Before predict must fit ！"
        assert X_predict.shape[1] == len(self.coefficient_), "The features of X_predict and coefficient_ must be the same !"
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self,X_predict):
        # >=0.5，预测为1(是)；<0.5，预测为0(否)
        return np.array(self.predict_probability(X_predict) >= 0.5, dtype=int)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "Logistic Regression"