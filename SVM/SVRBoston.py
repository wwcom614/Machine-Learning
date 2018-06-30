import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

# 线性SVM处理回归问题
from SVM.SVRFunction import StandardLinearSVR
svr = StandardLinearSVR()
svr.fit(X_train, y_train)
print("svr.score:",svr.score(X_test, y_test))
#svr.score: 0.6098690368116443