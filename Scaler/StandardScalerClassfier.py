import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        assert X.dim == 2, "Only support 2dim Array!"

        self.mean_ = np.array(np.mean(X[:,i]) for i in range(X.shape[1]))
        self.scale_ = np.array(np.std(X[:,i]) for i in range(X.shape[1]))

        return self

    def transform(self, X):
        assert X.dim == 2, "Only support 2dim Array!"
        assert self.mean_ is not None and self.scale_ is not None, "Before transform must fit !"
        assert X.shape[1] == len(self.mean_),"X的特征数必须与均值特征数相等"
        assert X.shape[1] == len(self.scale_),"X的特征数必须与标准差特征数相等"

        resultX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resultX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        return resultX


