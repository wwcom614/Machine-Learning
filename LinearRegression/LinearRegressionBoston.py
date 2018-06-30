import numpy as np
from sklearn import datasets


boston = datasets.load_boston()
X = boston.data
y = boston.target

#去除因y上限为50的错误数据
X = X[y < 50]
y = y[y < 50]

from Utils.TrainTestSplitFunction import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_ratio=0.2, seed=1)

# 基于自己写的正规方程解fit函数
from LinearRegression.LinearRegressionClassfier import LinearRegressionClassfier
normal_reg = LinearRegressionClassfier()
normal_reg.fit_normal(X_train, y_train)
#print("normal_reg.coefficient_:", normal_reg.coefficient_)
#print("normal_reg.interception_:", normal_reg.interception_)
print("normal_reg.score:", normal_reg.score(X_test, y_test))
# normal_reg.score: 0.7315154056267423

######################################

#梯度下降法之前，需要数据标准化(均值为0，方差为1，这样步长eta才好取)
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
###############################################################

# 基于自己写的批量梯度下降法fit函数
from LinearRegression.LinearRegressionClassfier import LinearRegressionClassfier
gd_reg = LinearRegressionClassfier()
gd_reg.fit_gd(X_train_standard, y_train)
#print("gd_reg.coefficient_:", gd_reg.coefficient_)
#print("gd_reg.interception_:", gd_reg.interception_)
print("gd_reg.score:", gd_reg.score(X_test_standard, y_test))
# gd_reg.score: 0.7314409236421784

######################################
# 基于自己写的随机梯度下降法fit函数
from LinearRegression.LinearRegressionClassfier import LinearRegressionClassfier
sgd_reg = LinearRegressionClassfier()
sgd_reg.fit_sgd(X_train_standard, y_train, max_iter=100)
#print("sgd_reg.coefficient_:", sgd_reg.coefficient_)
#print("sgd_reg.interception_:", sgd_reg.interception_)
print("sgd_reg.score:", sgd_reg.score(X_test_standard, y_test))


###########################################################################
# 基于自己写的批量梯度下降法(内部使用近似求导方式)fit函数
from GradientDescent.GradientDescentDebug import GradientDescentDebug
gd_debug_reg = LinearRegressionClassfier()
gd_debug_reg.fit_gd(X_train_standard, y_train)
#print("gd_reg.coefficient_:", gd_reg.coefficient_)
#print("gd_reg.interception_:", gd_reg.interception_)
print("gd_debug_reg.score:", gd_debug_reg.score(X_test_standard, y_test))
# gd_debug_reg.score: 0.7314409236421784


##################################
#scikit-learn的 LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scikit-learn的批量梯度下降法的线性回归
from sklearn.linear_model import LinearRegression
skt_lin_reg = LinearRegression()
skt_lin_reg.fit(X_train, y_train)

#print("skt_lin_reg.coef_:",skt_lin_reg.coef_)
#print("skt_lin_reg.intercept_:", skt_lin_reg.intercept_)
print("skt_lin_reg.score:", skt_lin_reg.score(X_test, y_test))
#lin_reg.score: 0.7578832841439207

#scikit-learn的随机梯度下降法的线性回归
from sklearn.linear_model import SGDRegressor
skt_sgd_reg = SGDRegressor(max_iter=100)
skt_sgd_reg.fit(X_train_standard, y_train)
print("skt_sgd_reg.score:", skt_sgd_reg.score(X_test_standard, y_test))

#############################################################

#线性回归具有强解释性的优点：
# label可以很好的与feature的数值很好的解释
# feature值的正负表示 正负影响；feature值的大小表示 影响大小
print(boston.feature_names[np.argsort(skt_lin_reg.coef_)])
#['NOX' 'DIS' 'PTRATIO' 'LSTAT' 'CRIM' 'INDUS' 'AGE' 'TAX' 'B' 'ZN' 'RAD' 'CHAS' 'RM']
#print(boston.DESCR)

#scikit-learn的 KNeighborsRegressor,默认未调优超参数
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)
print("knn_reg.score:",knn_reg.score(X_test, y_test))
#knn_reg.score: 0.47228892042824067

#scikit-learn的 KNeighborsRegressor,超参数调优
from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors" : [i for i in range(1,6)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1,6)],
        "p": [i for i in range(1,6)]
    }
]
knn_reg =  KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)
print("best_params_:", grid_search.best_params_)
#best_params_: {'n_neighbors': 4, 'p': 1, 'weights': 'distance'}
print("best_score_:", grid_search.best_score_)
#best_score_: 0.687420968580391
print("best_estimator_.score:",grid_search.best_estimator_.score(X_test,y_test))
#best_estimator_.score: 0.5857060526572068

