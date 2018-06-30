import numpy as np

np.random.seed(1)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = 0.5 * x**2 +x +2 + np.random.normal(0, 1, size=100)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

from Utils.PlotLearningCurve import plot_learning_curve

from sklearn.linear_model import LinearRegression
# 线性回归模型的学习曲线
plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)


from PolynomialRegression.PolynomialRidgeLassoRegression import PolynomialRegression
# 2阶多项式回归模型的学习曲线
poly2_reg = PolynomialRegression(degree=2)
plot_learning_curve(poly2_reg, X_train, X_test, y_train, y_test)