import numpy as np
from sklearn import datasets

digits = datasets.load_digits()

X = digits.data
y = digits.target.copy()

#Skewed data
y[digits.target==1] = 1
y[digits.target!=1] = 0


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)
print("log_reg.score:",log_reg.score(X_test, y_test))
#log_reg.score: 0.9777777777777777  过于偏执的数据方面，不可信

from Utils.Evaluate2Class import confusion_matrix
print("confusion_matrix:", confusion_matrix(y_test, y_predict))
'''
confusion_matrix: [[328   2]
 [  6  24]]
'''

from Utils.Evaluate2Class import precision_score
print("precision_score:", precision_score(y_test, y_predict))
#precision_score: 0.9230769230769231

from Utils.Evaluate2Class import recall_score
print("recall_score:", recall_score(y_test, y_predict))
#recall_score: 0.8

# 精准率precision_score和召回率recall_score的调和均值f1 score
from Utils.Evaluate2Class import f1_score
print("f1_score:", f1_score(y_test, y_predict))
#f1_score: 0.8571428571428571

#######################################################

#重要：decision_score，对每个样本的决策分数值
#默认分类评判标准逻辑回归decision_score的threshold是0，调整阈值threshold，用于均衡精准率和召回率
decision_score = log_reg.decision_function(X_test)

print("decision_score.shape:",decision_score.shape)
#decision_score.shape: (360,)

print("np.max(decision_score):",np.max(decision_score))
#np.max(decision_score): 12.636920028158261

print("np.min(decision_score):",np.min(decision_score))
#np.min(decision_score): -39.614726916569

y_predict2 = np.array(decision_score >= 5, dtype='int')


#scikit-learn
from sklearn.metrics import confusion_matrix
print("sklearn confusion_matrix:", confusion_matrix(y_test, y_predict2))
'''
sklearn confusion_matrix: [[330   0]
 [ 19  11]]
'''

from sklearn.metrics import precision_score
print("sklearn precision_score:", precision_score(y_test, y_predict2))
#sklearn precision_score: 1.0

from sklearn.metrics import recall_score
print("sklearn recall_score:", recall_score(y_test, y_predict2))
#sklearn recall_score: 0.36666666666666664

from sklearn.metrics import f1_score
print("sklearn f1_score:", f1_score(y_test, y_predict2))
#sklearn f1_score: 0.5365853658536585


################################################################
# 绘制随thresholds变化，精确率和召回率的变化曲线
import matplotlib.pyplot as plt

precisions = []
recalls = []
thresholds = np.arange(start=np.min(decision_score), stop=np.max(decision_score), step=0.1)
for threshold in thresholds:
    y_predict = np.array(decision_score >= threshold, dtype='int')
    precisions.append(precision_score(y_test, y_predict))
    recalls.append(recall_score(y_test, y_predict))

plt.plot(thresholds, precisions, label='precision')
plt.plot(thresholds, recalls, label='recall')
plt.legend()
plt.title("Thresholds -- Precision and Recall ")
plt.show()

# Precision-Recall曲线
plt.plot(precisions, recalls)
plt.title("Precision -- Recall")
plt.show()

# 绘制ROC曲线：FPR-TPR
from Utils.Evaluate2Class import TPR,FPR
fprs = []
tprs = []
thresholds = np.arange(start=np.min(decision_score), stop=np.max(decision_score), step=0.1)
for threshold in thresholds:
    y_predict = np.array(decision_score >= threshold, dtype='int')
    fprs.append(FPR(y_test, y_predict))
    tprs.append(TPR(y_test, y_predict))

# ROC(FPR-TPR)曲线
plt.plot(fprs, tprs)
plt.title("ROC(FPR-TPR)")
plt.show()



################################
#scikit-learn数据绘制随thresholds变化，精确率和召回率的变化曲线。
from sklearn.metrics import precision_recall_curve
#步长、起始threshold值，scikit-learn根据数据大小自己确定。
#thresholds的数量比precisions和recalls少1个，因为确定最大的threshold时，precision=1，recall=0
precisions, recalls, thresholds = precision_recall_curve(y_test, decision_score)

plt.plot(thresholds, precisions[:-1], label='precision')
plt.plot(thresholds, recalls[:-1], label='recall')
plt.legend()
plt.title("Sklearn Thresholds -- Precision and Recall ")
plt.show()

# scikit-learn数据绘制Precision-Recall曲线，面积越大，模型越好。找到陡峭下降那个点
plt.plot(precisions, recalls)
plt.title("Sklearn  Precision -- Recall")
plt.show()


# scikit-learn数据绘制ROC曲线：FPR-TPR
# TPR和FPR趋势一致;TPR越大越好，FPR越小越好
from sklearn.metrics import roc_curve
#步长、起始threshold值，scikit-learn根据数据大小自己确定。
fprs, tprs, thresholds = roc_curve(y_test, decision_score)
plt.plot(fprs, tprs)
plt.title("Sklearn  ROC(FPR-TPR)")
plt.show()

#roc_auc_score的auc是ROC曲线下方面积评分，对有偏数据不敏感(有偏数据还是要看PR曲线)
#roc_auc和ROC曲线主要用于看模型效果优劣，ROC曲线下方面积越大越好
from sklearn.metrics import roc_auc_score
print("roc_auc_score:",roc_auc_score(y_test, decision_score))
#roc_auc_score: 0.9968686868686869