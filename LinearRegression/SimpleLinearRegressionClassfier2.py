import numpy as np
from Utils.AccuracyFunction import r2_score

class SimpleLinearRegressionClassfier2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "SimpleLinearRegression only can process 1 dim data!"
        assert len(x_train) == len(y_train), "the size of x_train must be the same as y_train !"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        #向量化计算，使用向量点乘，对应 i 值的数值相乘，然后相加成1个数，性能比for循环高很多
        #分子
        nominator = (x_train - x_mean).dot(y_train - y_mean)

        #分母
        demoninator = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = nominator / demoninator
        self.b_ = y_mean - self.a_ * x_mean
        return self

    #待预测的数据可以多个，是个一维向量
    def predict(self, x_predict):
        assert x_predict.ndim == 1, "SimpleLinearRegression only can process 1 dim data!"
        assert self.a_ is not None and self.b_ is not None, "Before predict must fit!"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    #R方评测
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Simple Linear Regression 2"

