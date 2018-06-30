from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


#Boosting，模型间串行增强

##########################################################
#调用scikit-learn的集成学习：Ada Boosting
#Ada Boosting，第一个算法fit后，会降低大部分拟合样本点的权重，然后下个算法再次拟合
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
print("ada_clf.score:",ada_clf.score(X_test, y_test))
# ada_clf.score: 0.83

##########################################################
#调用scikit-learn的集成学习：Gradient Boosting
#Gradient Boosting，训练一个模型m1，产生错误e1；针对e1训练第2个模型m2,产生错误e2；针对e2训练第3个模型m3,产生错误e3...
#最终预测结果是m1+m2+m3+...
#Gradient Boosting自身就是以决策树为基础的
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=500)
gb_clf.fit(X_train, y_train)
print("gb_clf.score:",gb_clf.score(X_test, y_test))
# gb_clf.score: 0.87




