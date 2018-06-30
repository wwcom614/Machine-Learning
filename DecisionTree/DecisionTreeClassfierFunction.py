import numpy as np
from collections import Counter
from math import log

#y = [1,2,1,1,3,3]
#print(Counter(y))
#Counter({1: 3, 3: 2, 2: 1})

#p是y出现的概率，信息熵
def entropy(y):
    #将y转换成一个字典：key是y的原值，value是y的原值出现次数
    counter = Counter(y)
    result= 0.0
    for num in counter.values():
        p = num / len(y)
        result +=  -p * log(p)
    return result

#p是y出现的概率，基尼系数
def gini(y):
    #将y转换成一个字典：key是y的原值，value是y的原值出现次数
    counter = Counter(y)
    result= 1.0
    for num in counter.values():
        p = num / len(y)
        result -=  p**2
    return result

#找出在某个特征dim维度上，值<=value的索引数组 和 值>value的索引数组
def split(X, y, dim, value_mean):
    index_left = (X[:, dim] <= value_mean)
    index_right = (X[:, dim] > value_mean)
    return X[index_left], X[index_right], y[index_left], y[index_right]

#基于信息熵
def entropy_try_split(X, y):
    best_entropy = float('inf')
    best_dim = -1
    best_value = -1
    #遍历每个特征，也就是每个维度dim
    for dim in range(X.shape[1]):
        #在特征dim维度上排序每个X样本的索引存储到sorted_index，X本身不变
        sorted_index = np.argsort(X[:, dim])
        #遍历每个特征dim维度上的索引排序样本，找第i个和第i-1个均值，作为阈值划分值
        for i in range(1, len(X)):
            #如果两者相等，不再做划分
            if(X[sorted_index[i-1], dim] != X[sorted_index[i], dim]):
                value_mean = (X[sorted_index[i-1], dim] + X[sorted_index[i], dim]) / 2
                X_left, X_right, y_left, y_right = split(X, y, dim, value_mean)
                e = entropy(y_left) + entropy(y_right)
                if e < best_entropy:
                    best_entropy = e
                    best_dim = dim
                    best_value = value_mean
    return best_entropy, best_dim, best_value

#基于基尼系数
def gini_try_split(X, y):
    best_gini = float('inf')
    best_dim = -1
    best_value = -1
    #遍历每个特征，也就是每个维度dim
    for dim in range(X.shape[1]):
        #在特征dim维度上排序每个X样本的索引存储到sorted_index，X本身不变
        sorted_index = np.argsort(X[:, dim])
        #遍历每个特征dim维度上的索引排序样本，找第i个和第i-1个均值，作为阈值划分值
        for i in range(1, len(X)):
            #如果两者相等，不再做划分
            if(X[sorted_index[i-1], dim] != X[sorted_index[i], dim]):
                value_mean = (X[sorted_index[i-1], dim] + X[sorted_index[i], dim]) / 2
                X_left, X_right, y_left, y_right = split(X, y, dim, value_mean)
                g = gini(y_left) + gini(y_right)
                if g < best_gini:
                    best_gini = g
                    best_dim = dim
                    best_value = value_mean
    return best_gini, best_dim, best_value
