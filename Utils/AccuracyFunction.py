import numpy as np
from math import sqrt

def precision_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], "The size of true and predict must be the same !"
    return sum(y_true == y_predict)/ len(y_true)

#均方误差
def mse(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be the same as y_predict !"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

#均方根误差
def rmse(y_true, y_predict):
    return sqrt(mse(y_true, y_predict))

#平均绝对误差
def mae(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be the same as y_predict !"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

# R方误差：1 - 使用模型预测与真实值产生的均方误差/直接使用均值预测与真实值产生的均方误差，也就是方差
# var：表示方差，即(各项-均值)的平方求和/项数N ，std表示标准差，是var的平方根。
def r2_score(y_true, y_predict):
    return 1 - mse(y_true, y_predict)/np.var(y_true)
