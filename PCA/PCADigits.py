import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print("knn_clf.score:",knn_clf.score(X_test, y_test))
#knn_clf.score: 0.9944444444444445

from sklearn.decomposition import PCA
#降到n_components维，但这种方式不好，因为不知道要降到多少维合适，降维会造成与原矩阵偏差变大
#pca = PCA(n_components=2)

# 推荐使用这种方式，降维到原矩阵方差的95%，只丢失5%信息
pca = PCA(0.95)
pca.fit(X_train)
print("pca.components_.shape[0]:", pca.components_.shape[0])
# pca.components_.shape[0]: 28 ,PCA从64维 降到 28维 ，只丢失5%信息
print("(pca.explained_variance_ratio_:",pca.explained_variance_ratio_)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_clf_PCA = KNeighborsClassifier()
knn_clf_PCA.fit(X_train_reduction, y_train)
print("knn_clf_PCA.score:",knn_clf_PCA.score(X_test_reduction, y_test))
# knn_clf_PCA.score: 0.9944444444444445

#画图看下降维后的维度，与原矩阵的方差偏离度的曲线
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
# pca.explained_variance_ratio_是PCA后，每个特征与原矩阵的方差，0~1之间取值，值越接近1，与原矩阵偏差越小。
# PCA后的n_components值越接近原矩阵的特征数，pca.explained_variance_ratio_值越接近1
plt.plot([i for i in range(X_train.shape[1])], [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()


#PCA另外一个用途是，直接先降到二维，便于画图查看
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)

#手写识别总共10个数字, 10种颜色，2个维度
for i in range(10):
    plt.scatter(X_reduction[y == i, 0], X_reduction[y == i, 1], alpha=0.8)
plt.show()




