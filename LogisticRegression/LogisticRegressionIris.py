import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

#逻辑回归默认只能处理二分类问题，所以只取鸢尾花数据集前2种分类
#考虑方便可视化，所以只取鸢尾花数据集前2种特征
XX = X[y<2, :2]
yy = y[y<2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

from LogisticRegression.LogisticRegressionClassfier import LogisticRegressionClassfier
logistic_reg = LogisticRegressionClassfier()
logistic_reg.fit(XX, yy)
print("logistic_reg.score:", logistic_reg.score(XX, yy))

#################################################################################
# 决策边界
def x2(x1):
    return (- logistic_reg.interception_ - logistic_reg.coefficient_[0] * x1)/logistic_reg.coefficient_[1]

x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)


plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.plot(x1_plot, x2_plot, color='r', label='Decision Boundary')
plt.legend()
plt.show()


from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(logistic_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.legend()
plt.show()

# KNN分类的决策边界
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=30)
knn_clf.fit(iris.data[:,:2], iris.target)
plot_decision_boundary(knn_clf, axis=[4, 8, 1.5, 4.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.scatter(X[y==2, 0], X[y==2, 1], color='r', label='y==2')
plt.legend()
plt.show()

#######################################################################
#逻辑线性回归支持多分类- OVR
from sklearn.linear_model import LogisticRegression
log_reg_ovr = LogisticRegression(multi_class='ovr')
log_reg_ovr.fit(X_train[:,:2], y_train)
print("log_reg_ovr.score:",log_reg_ovr.score(X_test[:,:2],y_test))
#log_reg_ovr.score: 0.6

# 逻辑线性回归OVR多分类的决策边界
plot_decision_boundary(log_reg_ovr, axis=[4, 8, 1.5, 4.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.scatter(X[y==2, 0], X[y==2, 1], color='r', label='y==2')
plt.legend()
plt.show()


#OVR类，可以将任意二分类转换为多分类
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr.fit(X_train[:,:2], y_train)
print("ovr.score:",ovr.score(X_test[:,:2],y_test))
#ovr.score: 0.6


################################################################################


#逻辑线性回归支持多分类- OVO
from sklearn.linear_model import LogisticRegression
log_reg_ovo = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg_ovo.fit(X_train[:,:2], y_train)
print("log_reg_ovo.score:",log_reg_ovo.score(X_test[:,:2],y_test))
#log_reg_ovo.score: 0.8

# 逻辑线性回归OVO多分类的决策边界
plot_decision_boundary(log_reg_ovo, axis=[4, 8, 1.5, 4.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.scatter(X[y==2, 0], X[y==2, 1], color='r', label='y==2')
plt.legend()
plt.show()

#OVO类，可以将任意二分类转换为多分类
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
ovo = OneVsOneClassifier(lr)
ovo.fit(X_train[:,:2], y_train)
print("ovo.score:",ovo.score(X_test[:,:2],y_test))
#ovo.score: 0.6333333333333333