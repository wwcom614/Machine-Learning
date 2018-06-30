import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


boston = datasets.load_boston()

#print(boston.DESCR)
#506个样本，13个特征
#RM特征：average number of rooms per dwelling

print(boston.feature_names)
#RM索引在第5列，['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']

x = boston.data[:,5]  #只使用房间数量这个特征
y = boston.target

#看图发现样本上限被设定了y最大50，那么y=50的点是有误的，需要去除
x = x[y < 50]
y = y[y < 50]


from Utils.TrainTestSplitFunction import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_ratio=0.2, seed=1)

from LinearRegression.SimpleLinearRegressionClassfier2 import SimpleLinearRegressionClassfier2
reg = SimpleLinearRegressionClassfier2()
reg.fit(x_train, y_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color='r')
plt.show()

y_predict = reg.predict(x_test)

from Utils.AccuracyFunction import mse
#MSE评测,受y量纲影响
mse_test = mse(y_test, y_predict)
print("mse_test:",mse_test)


from Utils.AccuracyFunction import rmse
#RMSE评测，与y量纲相同
#RMSE对比MAE略好，其内部有平方，会将最大误差放大，所以如果RMSE小，说明各个误差都不大
rmse_test = rmse(y_test, y_predict)
print("rmse_test:",rmse_test)

from Utils.AccuracyFunction import mae
#MAE评测，与y量纲相同
mae_test = mae(y_test, y_predict)
print("mae_test:",mae_test)

from Utils.AccuracyFunction import r2_score
#R方评测，推荐使用
r2_test = r2_score(y_test, y_predict)
print("r2_test:", r2_test)

#封装在线性回归类中的R方评测，注意参数是测试样本和测试label
print("reg.score", reg.score(x_test, y_test))


from sklearn.metrics import mean_squared_error
print("sklearn.metrics mean_squared_error:", mean_squared_error(y_test, y_predict))

from sklearn.metrics import mean_absolute_error
print("sklearn.metrics mean_absolute_error:", mean_absolute_error(y_test, y_predict))

from sklearn.metrics import r2_score
print("sklearn.metrics r2_score:", r2_score(y_test, y_predict))

