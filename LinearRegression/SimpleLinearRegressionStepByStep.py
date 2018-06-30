import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,3.,2.,3.,5.])

x_mean = np.mean(x)
y_mean = np.mean(y)

nominator = 0.0 #分子
denominator = 0.0 #分母

for x_i, y_i in zip(x, y):
    nominator += (x_i - x_mean) * (y_i - y_mean)
    denominator += (x_i - x_mean) ** 2

a = nominator / denominator
b = y_mean - a * x_mean

print("a=", a)
print("b=", b)

y_hat = a * x + b

plt.scatter(x, y)
plt.plot(x, y_hat, color ='r')
plt.axis([0,6,0,6])
plt.show()


x_predict = 6
y_predict1 = a * x_predict + b
print("y_predict1=", y_predict1)


from LinearRegression.SimpleLinearRegressionClassfier1 import SimpleLinearRegressionClassfier1
reg1 = SimpleLinearRegressionClassfier1()
reg1.fit(x, y)

y_predict2 = reg1.predict(np.array([x_predict]))
print("y_predict2=", y_predict2)
print("reg1.a_=", reg1.a_)
print("reg1.b_=", reg1.b_)

from LinearRegression.SimpleLinearRegressionClassfier2 import SimpleLinearRegressionClassfier2
reg2 = SimpleLinearRegressionClassfier2()
reg2.fit(x, y)

y_predict3 = reg1.predict(np.array([x_predict]))
print("y_predict3=", y_predict3)
print("reg1.a_=", reg1.a_)
print("reg1.b_=", reg1.b_)


