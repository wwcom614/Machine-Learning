# SVM处理回归问题

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 使用标准线性SVR
from sklearn.svm import LinearSVR
def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("linearSVR", LinearSVR(epsilon=epsilon))
    ])


