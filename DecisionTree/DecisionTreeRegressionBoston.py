import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
print("dt_reg.score:",dt_reg.score(X_test, y_test))
#dt_reg.score: 0.5232578533118353  未调参，泛化效果不好

