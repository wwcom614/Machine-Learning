from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)


##########################################################
#scikit-learn随机森林分类器
from sklearn.ensemble import RandomForestClassifier
#随机森林中，有n_estimators颗决策树
#每颗决策树最多有max_leaf_nodes个节点
#随机森林是Bagging Decision Trees，且决策树在节点划分上，在随机的特征子集上寻找最优划分特征
rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, max_leaf_nodes=16, oob_score=True, n_jobs=-1)
rf_clf.fit(X, y)
print("rf_clf.oob_score_:",rf_clf.oob_score_)
#rf_clf.oob_score_: 0.92


##########################################################
#scikit-learn的Extra Trees分类器
from sklearn.ensemble import ExtraTreesClassifier
#Extra Trees中，有n_estimators颗决策树
#Extra Trees默认是不放回取样pasting，需要显示指定为放回取样bagging，bootstrap=True
#Extra Trees是Bagging Decision Trees，而且决策树在节点划分上，使用随机的特征和随机的阈值。提供了额外的随机性，抑制了过拟合，但增大了偏差
et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=666)
et_clf.fit(X, y)
print("et_clf.oob_score_:",et_clf.oob_score_)
#et_clf.oob_score_: 0.892






