from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

##########################################################
#样本产生差异化的方法1：样本随机放回取样
#调用scikit-learn的集成学习：放回取样bagging(bootstrap=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
# 基于决策树算法做bagging
#决策树是非参机器学习，因为其内部有很多的超参数、剪枝等，将更能产生出差异较大的子模型，更加随机。
#集成学习非常适合使用这种算法。如果需要集成成百上千子模型的时候，首选决策树
#n_estimators，集成多少个指定的模型作为子模型
#max_samples，每个子模型使用多少个样本数据
#bootstrap=True 放回取样bagging； bootstrap=False 不放回取样pasting
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=500, max_samples=100, bootstrap=True)
bagging_clf.fit(X_train, y_train)
print("bagging_clf.score:",bagging_clf.score(X_test, y_test))
# bagging_clf.score: 0.9

##########################################################
#样本产生差异化的方法1：样本随机放回取样改良：更合理的最大化利用有限的样本数据的方法
#调用scikit-learn的集成学习：放回取样bagging(bootstrap=True,oob_score=True)
#放回取样会导致一部分样本很可能从未被取到。数学概率计算平均大约有37%的样本没有取到
#那么就可以考虑fit使用全量数据训练拟合，使用未取到的样本OOB(out of bagging)作为测试数据集验证效果
#oob_score默认为False，不记录未取到的样本。显示指定oob_score=True，将记录未取到的样本
#Bagging的思路极易使用并行化处理，scikit-learn提供并行执行核数的参数n_jobs，如果配置n_jobs=-1，将使用所有的核
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True, n_jobs=1)
bagging_clf.fit(X, y)
print("bagging_clf.oob_score_:",bagging_clf.oob_score_)
# bagging_clf.oob_score_: 0.918


##########################################################
#样本产生差异化的方法2：不随机取样本，而是随机取特征random_subspaces
#max_features，每次最大随机取样本的max_features个特征，只取部分
#max_samples置为与样本总量一样，那么就是不对样本随机取样，每次都是取全量 + 随机取特征random_subspaces(bootstrap_features=True)
random_subspaces_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=500,
                                bootstrap=True, oob_score=True,
                                max_features=1, bootstrap_features=True)
random_subspaces_clf.fit(X, y)
print("random_subspace_clf.oob_score_:",random_subspaces_clf.oob_score_)
# random_subspace_clf.oob_score_: 0.82


##########################################################
#样本产生差异化的方法3：既随机取样本，又随机取特征random_patches
#max_features，每次随机取样本的max_features个特征，只取部分
#max_samples随机放回取样 + 随机取特征random_subspaces(bootstrap_features=True)
random_patches_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                         bootstrap=True, oob_score=True,
                                         max_features=1, bootstrap_features=True)
random_patches_clf.fit(X, y)
print("random_patches_clf.oob_score_:",random_patches_clf.oob_score_)
# random_patches_clf.oob_score_: 0.854




