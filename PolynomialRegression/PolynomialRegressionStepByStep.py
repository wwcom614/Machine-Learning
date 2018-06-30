import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = 0.5 * x**2 +x +2 + np.random.normal(0, 1, size=100)


# 使用线性回归尝试拟合，效果不好
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict1 = lin_reg.predict(X)



# 给现有数据集增维，增加特征，提升拟合效果
X2 = np.hstack([X**2, X])
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
print("lin_reg2.coef_:",lin_reg2.coef_)
#lin_reg2.coef_: [0.5357447  0.95253067]
print("lin_reg2.intercept_:",lin_reg2.intercept_)
#lin_reg2.intercept_: 1.9680598124448996


from sklearn.preprocessing import PolynomialFeatures
#添加几次幂的特征。
poly = PolynomialFeatures(degree=2)
poly.fit(X)
X3 = poly.transform(X)
# 第1列是样本值0次幂，第2列是样本值1次幂，第3列是样本值2次幂
# 遇上更多次幂，会比这复杂。
# 3次幂会生成6列(1,X1,X2,X1方，X2方，X1*X2)，4次幂会生成10列
# 多次幂会使样本相应特征变大，所以需要再进行标准化scaler
print(X3[:3,:])
'''
[[ 1.         -0.49786797  0.24787252]
 [ 1.          1.32194696  1.74754377]
 [ 1.         -2.99931375  8.99588298]]
'''
lin_reg3 = LinearRegression()
lin_reg3.fit(X3, y)
y_predict3 = lin_reg3.predict(X3)
print("lin_reg3.coef_:",lin_reg3.coef_)
#lin_reg3.coef_: [0.         0.95253067 0.5357447 ]
print("lin_reg3.intercept_:",lin_reg3.intercept_)
#lin_reg3.intercept_: 1.9680598124449005

#将样本增维、标准化、线性拟合综合到一起，使用pipeline，已将其封装成函数
from PolynomialRegression.PolynomialRidgeLassoRegression import PolynomialRegression
poly_reg = PolynomialRegression(degree=2)
poly_reg.fit(X, y)
y_predict4 = poly_reg.predict(X)



plt.scatter(x, y)
plt.plot(x, y_predict1, color='r')
#plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='g')
#plt.plot(np.sort(x), y_predict3[np.argsort(x)], color='y')
plt.plot(np.sort(x), y_predict4[np.argsort(x)], color='g')
plt.show()