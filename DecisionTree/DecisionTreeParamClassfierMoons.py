import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(noise=0.25, random_state=666)

from sklearn.tree import DecisionTreeClassifier
#超参数1：max_depth，决策树深度，越小越不容易过拟合
dt_clf1 = DecisionTreeClassifier(max_depth=2)
dt_clf1.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(dt_clf1, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.title("max_depth=2")
plt.show()


#超参数2：min_samples_split，划分结束判断条件。如果只剩min_samples_split个样本，则不再继续划分。越大越不容易过拟合
dt_clf2 = DecisionTreeClassifier(min_samples_split=10)
dt_clf2.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.title("min_samples_split=10")
plt.show()

#超参数3：min_samples_leaf，划分结束判断条件。最底层的叶子节点需要至少保留min_samples_leaf个样本，则不再继续划分。越大越不容易过拟合
dt_clf2 = DecisionTreeClassifier(min_samples_leaf=6)
dt_clf2.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.title("min_samples_leaf=6")
plt.show()


#超参数4：max_leaf_nodes，划分结束判断条件。最多有max_leaf_nodes个叶子节点，则不再继续划分。越小越不容易过拟合
dt_clf2 = DecisionTreeClassifier(max_leaf_nodes=4)
dt_clf2.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.title("max_leaf_nodes=4")
plt.show()
