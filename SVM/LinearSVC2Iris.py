import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

#SVM处理二分类问题，所以只取鸢尾花数据集前2种分类
#考虑方便可视化，所以只取鸢尾花数据集前2种特征
XX = X[y<2, :2]
yy = y[y<2]

#涉及距离，所以和KNN一样，使用SVM之前，务必需要做数据标准化处理！使得各个维度单位一致
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(XX)
X_standard = standardScaler.transform(XX)

#基于SVM算法处理线性分类问题
from sklearn.svm import LinearSVC
#超参数C，正则是为了容错.C越大，yita越小，容错空间越小。
# C无穷，yita=0，就变成Hard Margin SVM了
#例如，Hard Margin SVM: C=1e9
#例如，Soft Margin SVM: C=0.01
svc = LinearSVC(C=1e9, multi_class='ovr', penalty='l2')
svc.fit(X_standard, yy)

#################################################################################
# 绘制决策边界
from Utils.PlotDecisionBoundary import plot_svc_decision_boundary
plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[yy==0, 0], X_standard[yy==0, 1], color='g', label='yy==0')
plt.scatter(X_standard[yy==1, 0], X_standard[yy==1, 1], color='b', label='yy==1')
plt.legend()
plt.show()

