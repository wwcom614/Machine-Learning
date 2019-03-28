import numpy as np

# 真实值1，预测值1
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict), "真实数据和预测数据size需要一致！"
    return np.sum((y_true == 1) & (y_predict == 1))

# 真实值0，预测值0
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict), "真实数据和预测数据size需要一致！"
    return np.sum((y_true == 0) & (y_predict == 0))

# 真实值0，预测值1
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict), "真实数据和预测数据size需要一致！"
    return np.sum((y_true == 0) & (y_predict == 1))

# 真实值1，预测值0
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict), "真实数据和预测数据size需要一致！"
    return np.sum((y_true == 1) & (y_predict == 0))


def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true,y_predict), FP(y_true,y_predict)],
        [FN(y_true,y_predict), TP(y_true,y_predict)]
    ])

# 精准率Precision
# 分子TP是真实值为1，预测值为1
# 分母是 分子TP + FP预测值为1但真实值为0，也就是该类别预测值为1的全部数量
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0

# 召回率Recall
# 分子TP是真实值为1，预测正确1
# 分母是 分子TP + FN预测值为0但真实值为1，也就是该类别真实值为1的全部数量
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2 * precision * recall /(precision + recall)
    except:
        return 0.0

# TPR就是召回率Recall
# 分子TP是真实值为1，预测正确1
# 分母是 分子TP + FN预测值为0但真实值为1，也就是该类别真实值为1的全部数量
def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

# FPR
# 分子FP预测值为1，但真实值为0
# 分母是 分子FP + TN预测值为0,真实值为0，也就是该类别真实值为0的全部数量
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0