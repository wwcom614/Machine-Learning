import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.random.normal(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.metrics import mean_squared_error

from PolynomialRegression.PolynomialRidgeLassoRegression import PolynomialRegression
poly20_reg = PolynomialRegression(degree=20)
poly20_reg.fit(X_train, y_train)
y_poly20_predict = poly20_reg.predict(X_test)
print("poly20 MSE:", mean_squared_error(y_poly20_predict, y_test))
# poly20 MSE: 1935508.353986661

from PolynomialRegression.PolynomialRidgeLassoRegression import RidgeRegression
ridge20_reg = RidgeRegression(degree=20, alpha=0.0001)
ridge20_reg.fit(X_train, y_train)
y_ridge20_predict = ridge20_reg.predict(X_test)
print("ridge20 MSE:", mean_squared_error(y_ridge20_predict, y_test))
# ridge20 MSE: 0.9822376458926243

from PolynomialRegression.PolynomialRidgeLassoRegression import LassoRegression
lasso20_reg = LassoRegression(degree=20, alpha=0.01)
lasso20_reg.fit(X_train, y_train)
y_lasso20_predict = lasso20_reg.predict(X_test)
print("lasso20 MSE:", mean_squared_error(y_lasso20_predict, y_test))
#lasso20 MSE: 1.0896170543531627

X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
plt.scatter(x, y)
plt.plot(X_plot[:, 0], poly20_reg.predict(X_plot), color='r', label='polynomial')
plt.plot(X_plot[:, 0], ridge20_reg.predict(X_plot), color='y', label='ridge')
plt.plot(X_plot[:, 0], lasso20_reg.predict(X_plot), color='g', label='lasso')
plt.legend()
plt.axis([-3,3,0,6])
plt.show()