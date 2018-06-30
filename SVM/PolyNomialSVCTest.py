import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
#默认100个样本，2个特征，可配置
X, y = datasets.make_moons(noise=0.15,random_state=666)

#使用多项式+线性SVM
from SVM.SVCFunction import PolynomialSVC
poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.legend()
plt.show()

#使用多项式核函数的SVM
from SVM.SVCFunction import PolynomialKernalSVC
poly_kernal_svc = PolynomialKernalSVC(degree=3)
poly_kernal_svc.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(poly_kernal_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.legend()
plt.show()


#使用高斯RBF核函数的SVM
#gamma越大，模型复杂度越高，越高窄(紧贴某类数据集，有可能过拟合)
#gamma越小，模型复杂度越低，越胖宽(极小时会和线性差不多，有可能欠拟合)
from SVM.SVCFunction import RBFKernalSVC
rbf_svc = RBFKernalSVC(gamma=1.0)
rbf_svc.fit(X, y)

from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(rbf_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1], color='g', label='y==0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', label='y==1')
plt.legend()
plt.show()