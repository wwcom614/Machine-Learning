'''
KNN:  K Nearest Neighbours
优点：
思想极简
所需应用数学知识近乎为零
效果好
可以解决分类问题
也可以解决回归问题


缺点：
计算量大
离预测点近的数据正确性会严重影响预测结果
预测结果不可解释
维度灾难(可能需要考虑降维)

'''

import numpy as np
import matplotlib.pyplot as plt


##############################################
# 构造样本数据

raw_data_X = [[3.393533211,2.331273381],
              [3.110073483,1.781539638],
              [1.343808831,3.368360954],
              [3.582294042,4.679179110],
              [2.280362439,2.866990263],
              [7.423436942,4.696522875],
              [5.745051997,3.533989803],
              [9.172168622,2.511101045],
              [7.792786481,3.424088941],
              [7.939820817,0.791637231]]

raw_data_y = [0,0,0,0,0,1,1,1,1,1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 构造预测记录
predictx = np.array([8.093607318,3.365731514])

plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color = 'g', label = "good")
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color = 'r', label = "bad")
plt.scatter(predictx[0], predictx[1], color = 'b', label = "predict" )
plt.legend()
plt.show()

##############################################
# KNN算法

from math import sqrt

K = 6
distances = []
for x_train in X_train:
    #欧拉距离
    d = sqrt(np.sum((x_train - predictx) ** 2))
    #distance数组记录各个样本点与预测点的欧拉距离
    distances.append(d)

#上述for循环可以简化为一行
#distance = [sqrt(np.sum((x_train-predictx)**2)) for x_train in X_train]

#值从小到大排序，并直接获取到排序后值的索引值
nearestIndex = np.argsort(distances)

#topK_y = [y_train[i] for i in nearestIndex[:K]]
topK_y = y_train[nearestIndex[:K]]
print(topK_y)
#[1 1 1 1 1 0]

#统计
from collections import Counter
votes = Counter(topK_y)
print(votes)
#Counter({1: 5, 0: 1})

print(votes.most_common(1))
#[(1, 5)]

#预测结果
predicty = votes.most_common(1)[0][0]
print("predicty:",predicty)
#  1

from KNN.KNNFunction import KNN_Fuction

f = KNN_Fuction(K, X_train, y_train, predictx)
print("f:",f)

