import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data[:, 2:]
y = iris.target

#信息熵entropy的计算比基尼系数gini稍慢，所以sklearn默认criterion="gini"，大多数时候两者没有区别
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_clf.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0, 0], X[y==0, 1], label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], label='y==1')
plt.scatter(X[y==2, 0], X[y==2, 1], label='y==2')
plt.legend()
plt.show()


#基于自己写的决策树函数做划分
from DecisionTree.DecisionTreeClassfierFunction import try_split,split,entropy

#第1次遍历划分
best_entropy1, best_dim1, best_value1 = try_split(X, y)
print("best_entropy1:",best_entropy1)
#best_entropy1: 0.6931471805599453
print("best_dim1:",best_dim1)
#best_dim1: 0
print("best_value1:",best_value1)
#best_value1: 2.45
X1_left, X1_right, y1_left, y1_right = split(X, y, best_dim1, best_value1)
print("entropy(y1_left):", entropy(y1_left))
#entropy(y1_left): 0.0  ，OK
print("entropy(y1_right):", entropy(y1_right))
#entropy(y1_right): 0.6931471805599453


#第2次遍历划分
best_entropy2, best_dim2, best_value2 = try_split(X1_right, y1_right)
print("best_entropy2:",best_entropy2)
#best_entropy2: 0.4132278899361904
print("best_dim2:",best_dim2)
#best_dim2: 1
print("best_value2:",best_value2)
#best_value2: 1.75
X2_left, X2_right, y2_left, y2_right = split(X, y, best_dim2, best_value2)
print("entropy(y2_left):", entropy(y2_left))
#entropy(y2_left): 0.8525876833625409
print("entropy(y2_right):", entropy(y2_right))
#entropy(y2_right): 0.10473243910508653


