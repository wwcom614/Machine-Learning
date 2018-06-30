import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression.PolynomialLogisticRegression import PolynomialLogisticRegression

######################################################
#数据
np.random.seed(666)
X = np.random.normal(0, 1, size=(200,2))
y = np.array(X[:,0] ** 2 + X[:,1]  < 1.5 ,dtype=int)
#加噪音
for _ in range(20):
    y[np.random.randint(200)] = 1

#区分训练数据集和测试数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

#######################################################################
#自己写的多项式逻辑回归训练和评分
poly_logistic_reg = PolynomialLogisticRegression(degree=20)
poly_logistic_reg.fit(X_train, y_train)
print("poly_logistic_reg.score:", poly_logistic_reg.score(X_test, y_test))
#poly_logistic_reg.score: 0.9

#自己写的多项式逻辑回归绘制决策边界
from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(poly_logistic_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

#####################################################################################
#scikit-learn逻辑回归
from sklearn.linear_model import LogisticRegression
skl_logistic_reg = LogisticRegression()
skl_logistic_reg.fit(X_train, y_train)
print("skl_logistic_reg.score:",skl_logistic_reg.score(X_test, y_test))
#skl_logistic_reg.score: 0.875

#scikit-learn逻辑回归绘制决策边界
from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(skl_logistic_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

##############################################################################################################
#scikit-learn多项式逻辑回归
from LogisticRegression.PolynomialLogisticRegression import SKLPolynomialLogisticRegression
skl_poly_logistic_reg = SKLPolynomialLogisticRegression(degree=20, C=0.1, penalty='l1')
skl_poly_logistic_reg.fit(X_train, y_train)
print("skl_poly_logistic_reg.score:", skl_poly_logistic_reg.score(X_test, y_test))
#skl_poly_logistic_reg.score: 0.85

#scikit-learn多项式逻辑回归绘制决策边界
from Utils.PlotDecisionBoundary import plot_decision_boundary
plot_decision_boundary(skl_poly_logistic_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()