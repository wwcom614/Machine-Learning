import numpy as np

from sklearn.model_selection import train_test_split  #样本拆分为训练集和测试集
from sklearn.neighbors import KNeighborsClassifier # KNN算法
from sklearn.metrics import accuracy_score  #准确率
from sklearn.model_selection import GridSearchCV  #网格搜索调优超参数

from sklearn import datasets

X = datasets.load_digits().data
y = datasets.load_digits().target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
#自己编写网格搜索，超参数调优：weights 和 k
# 是否考虑距离加权参数(可解决平票问题，有可能考虑距离更合理.
# weights=uniform不考虑距离(默认值), weights=distance考虑距离
best_method_1 = ""
best_score_1 = 0.0
best_k_1 = -1
for method in ["uniform", "distance"]:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score_1:
            best_method_1 = method
            best_score_1 = score
            best_k_1 = k

print("best score_1:", best_score_1)
# best score_1: 0.9916666666666667
print("best_k_1:", best_k_1)
# best_k_1: 2  注：如果k的值在range最大临界值附近，调整range最大值范围再尝试
print("best_method_1:", best_method_1)
# best_method_1: uniform
'''

'''
#自己编写网格搜索，超参数调优：weights=distance的前提下，p 和 k
# 考虑距离的前提下(weights="distance"). 距离的算法类型，p=1曼哈顿距离，p=2欧拉距离(默认值)
# ,p距离公式明可夫斯基距离，那种距离算法更合适？
best_p_2 = 0
best_score_2 = 0.0
best_k_2 = -1
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score_2:
            best_score_2 = score
            best_k_2 = k
            best_p_2 = p

print("best score_2:", best_score_2)
# best score_2: 0.9916666666666667
print("best_k_2:", best_k_2)
# best_k_2: 1。注：如果k的值在range最大临界值附近，调整range最大值范围再尝试
print("best_p_2:", best_p_2)
# best_p_2: 3
'''


#scikit-learn的网格搜索 超参数调优
#param_grid列表，里面每个字典都是一个分类器
#网格搜索 超参数字典，key是超参数，value是个超参数的候选调优值列表


param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]#包含两组参数



knn_clf = KNeighborsClassifier()
#param_grid内部有几个分类器字典，并行n_jobs个CPU核来处理。n_jobs=-1是指所有核
#verbose值越大，中间输出结果越详细
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)
#名字后面有下划线的，是指的机器计算出来的参数，而不是人定义的参数

print(grid_search.best_score_)
#0.988169798190675
print(grid_search.best_params_)
#{'n_neighbors': 1, 'p': 3, 'weights': 'distance'}

knn_clf = grid_search.best_estimator_
knn_clf.score(X_test, y_test)