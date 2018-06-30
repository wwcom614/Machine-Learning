import numpy as np

from sklearn import datasets
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


#算法1fit
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
print("log_clf.score:",log_clf.score(X_test, y_test))
#log_clf.score: 0.85

#算法2fit
from sklearn.svm import SVC
svc_clf = SVC()
svc_clf.fit(X_train, y_train)
print("svc_clf.score:",svc_clf.score(X_test, y_test))
#svc_clf.score: 0.87

#算法3fit
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
print("dt_clf.score:",dt_clf.score(X_test, y_test))
#dt_clf.score: 0.83

##########################################################
#自己写个简单的让3种算法投票集成学习
y_log_predict = log_clf.predict(X_test)
y_svc_predict = svc_clf.predict(X_test)
y_dt_predict = dt_clf.predict(X_test)

y_predict = np.array((y_log_predict + y_svc_predict + y_dt_predict) >= 2, dtype='int')

from sklearn.metrics import accuracy_score
print("accuracy_score:",accuracy_score(y_test, y_predict))
#accuracy_score: 0.85
##########################################################

##########################################################
#调用scikit-learn的集成学习：投票hard voting
from sklearn.ensemble import VotingClassifier
hard_voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svc_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))
], voting='hard') #少数服从多数就是hard

hard_voting_clf.fit(X_train, y_train)
print("hard_voting_clf.score:",hard_voting_clf.score(X_test, y_test))
#hard_voting_clf.score: 0.86

##########################################################
#调用scikit-learn的集成学习：概率权重投票soft voting
#soft voting比hard votng更合理，soft voting要求集合的每一个模型都能估计概率predict_proba，
# 也就是分给某个类的概率是多少
#KNN算法如果不考虑权重，概率predict_proba=投票最多数/投票总数
#KNN算法如果考虑权重，概率predict_proba=投票最多数点的权重和/总权重和
#决策树算法和KNN算法一样，计算的是叶子节点中的概率predict_proba
#SVM默认概率predict_proba是关闭的，需要probability=True打开，但计算需要计算资源和时间
from sklearn.ensemble import VotingClassifier
soft_voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svc_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666))
], voting='soft') #少数服从多数就是hard

soft_voting_clf.fit(X_train, y_train)
print("soft_voting_clf.score:",soft_voting_clf.score(X_test, y_test))
#soft_voting_clf.score: 0.88