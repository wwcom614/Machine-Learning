import numpy as np
from math import sqrt
from collections import Counter
from Utils.AccuracyFunction import precision_score

class KNNClassfier:
    #初始化分类器
    def __init__(self,k):
        assert 1 <= k , "K must be >= 1!"
        self.k = k
        #私有成员变量加下划线
        self._X_train = None
        self._y_train = None

    #训练分类器
    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be the same as y_train!"
        assert self.k <= X_train.shape[0], "K must be  <= the size of X_train!"

        #训练分类器。KNN算法不用训练，模型就是样本数据集
        self._X_train = X_train
        self._y_train = y_train
        #在KNN中可以不返回，但按scikit-learn标准是返回self
        return self

    #用训练好的分类器预测
    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, "must be fit before predict!"
        assert self._X_train.shape[1] == X_predict.shape[1], "The features of  X_train must be the same as predictx"

        #调用私有的预测函数，返回的转换为scikit-learn需要的数组
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    #私有的预测函数定义
    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1], "The features of  X_train must be the same as predict x !"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearestIndex = np.argsort(distances)

        topK_y = self._y_train[nearestIndex[:self.k]]

        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    #准确度效果评估
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return precision_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k = %d)" % self.k


