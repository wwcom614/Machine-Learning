# SVM处理分类问题

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# 使用传统多项式的线性SVM
def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("linearSVC", LinearSVC(C=C))
    ])


# 使用多项式核函数的SVM
from sklearn.svm import SVC
def PolynomialKernalSVC(degree, C=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("kernalSVC", SVC(kernel="poly", degree=degree, C=C))
    ])

# 使用高斯RBF核函数的SVM
def RBFKernalSVC(gamma=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=gamma))
    ])
