from sklearn.cluster import KMeans
import numpy as np

# 特征矩阵X
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# 标记矩阵y
y = [0, 0, 0, 1, 1, 1]

# Kmeans算法拟合训练，2个中心点(分类)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X, y)

# 对测试集预测
print(kmeans.predict([[0, 0], [4, 4]]))
# [0 1]