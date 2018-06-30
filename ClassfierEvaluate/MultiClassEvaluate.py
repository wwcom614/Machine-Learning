import numpy as np
import matplotlib.pyplot as plt


from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)
print("log_reg.score:",log_reg.score(X_test, y_test))
#log_reg.score: 0.95

from sklearn.metrics import precision_score
print("precision_score:",precision_score(y_test, y_predict, average="micro"))
#precision_score: 0.95

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(y_test, y_predict)
print("confusion_matrix:\n",cfm)
'''
confusion_matrix:
 [[37  0  0  0  0  1  0  0  0  0]
 [ 0 28  0  0  0  0  0  0  2  0]
 [ 0  0 34  1  0  0  0  0  0  0]
 [ 0  0  0 40  0  1  0  0  3  0]
 [ 0  1  0  0 41  0  0  0  0  0]
 [ 0  0  0  1  0 28  0  0  1  0]
 [ 0  0  0  0  0  1 29  0  0  0]
 [ 0  0  0  0  0  0  0 35  0  1]
 [ 0  1  0  0  0  0  0  0 38  0]
 [ 0  1  0  2  0  0  0  0  1 32]]
'''
#绘制混淆矩阵。越亮数量越多
#plt.matshow(cfm, cmap=plt.cm.gray)
#plt.show()

#但关注成功数没用，想看错误数，所以需要绘制混淆矩阵中非对角线的错误预测矩阵
row_sums = np.sum(cfm, axis=1)
err_matrix = cfm / row_sums
#将err_matrix对角线数字置0
np.fill_diagonal(err_matrix, 0)
#绘制错误矩阵。越亮数量越多。需要微调相应的threshold，以及看看相应的样本数据是否有问题
plt.matshow(err_matrix, cmap=plt.cm.gray)
plt.show()