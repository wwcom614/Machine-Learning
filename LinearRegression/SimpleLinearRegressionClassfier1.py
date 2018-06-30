import numpy as np

class SimpleLinearRegressionClassfier1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "SimpleLinearRegression only can process 1 dim data!"
        assert len(x_train) == len(y_train), "the size of x_train must be the same as y_train !"

        nominator = 0.0 #分子
        demoninator = 0.0 #分母

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        for x_i, y_i in zip(x_train, y_train):
            nominator += (x_i - x_mean) * (y_i - y_mean)
            demoninator += (x_i - x_mean) ** 2

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

    def __repr__(self):
        return "Simple Linear Regression 1"

