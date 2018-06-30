import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

#print(digits.DESCR)
#5620个样本(实际只有1797个)，每个样本64个特征(8*8像素图像，每个像素0~16随机灰度值)

print(digits.data.shape)
#(1797, 64)

X = digits.data
y = digits.target

#找666行样本画图看下
#digit_image_666 = X[666,:].reshape(8,8)
#plt.imshow(digit_image_666, cmap = matplotlib.cm.binary)
#plt.show()

from Utils.TrainTestSplitFunction import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_ratio=0.2)

from KNN.KNNClassfier import KNNClassfier

#构造函数
knn_clf = KNNClassfier(k = 3)

#训练(拟合)
knn_clf.fit(X_train, y_train)

#预测
#y_predict = knn_clf.predict(X_test)

#准确率效果评估
print("knn_clf.score:", knn_clf.score(X_test, y_test))
# knn_clf.score:98.88579387186628

#使用scikit-learn尝试
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

y_predict = knn_clf.predict(X_test)

from sklearn.metrics import accuracy_score

print("sklearn baccuracy_score:", accuracy_score(y_test, y_predict))
# sklearn accuracy_score:0.9944444444444445

#使用交叉验证, 默认分3份，cv参数可以指定分几份
from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier()
print("cross_val_score:" ,cross_val_score(knn_clf, X_train, y_train, cv=5))
# cross_val_score: [0.97945205 0.97241379 0.98601399 0.98596491 0.98943662]

# 基于交叉验证做超参数调优，不 过拟合，更可信。 缺点：慢了分组数量K倍
# 交叉验证默认将训练数据集分为3份，2份做训练，1份做验证，进行三组
# scikit-learn中的网格搜索GridSearchCV中的CV就是指的交叉验证Cross Validation
best_score, best_k, best_p = 0., 0., 0.

for k in range(2,11):
    for p in range(1,6):
        knn_clf = KNeighborsClassifier(weights="distance",n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train, cv=3)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = p

print("best_score:",best_score)
# best_score: 0.9805275188324448
print("best_k:",best_k)
# best_k: 4
print("best_p:",best_p)
# best_p: 2

best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=best_k, p=best_p)
best_knn_clf.fit(X_train, y_train)
print("best_knn_clf.score:",best_knn_clf.score(X_test, y_test))
# best_knn_clf.score: 0.9916666666666667

# GridSearchCV也是基于交叉验证CV
from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1,11)],
        'p': [i for i in  range(1,6)]
    }
]
grid_search = GridSearchCV(knn_clf, param_grid, verbose=1, cv=3)
grid_search.fit(X_train, y_train)
print("grid_search.best_params_:",grid_search.best_params_)

print("grid_search.best_score_:",grid_search.best_score_)

best_grid_knn_clf = grid_search.best_estimator_

print("best_grid_knn_clf.score:",best_grid_knn_clf.score(X_test, y_test))
