import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def plot_learning_curve(ML, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    #学习曲线：随训练样本数的增多，训练数据集的预测RMSE和测试数据集的预测RMSE的趋势图，可以直观看出过拟合和欠拟合情况
    for i in range(1, len(X_train)+1):
        ML.fit(X_train[:i], y_train[:i])

        y_train_predict = ML.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict[:i]))

        y_test_predict = ML.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train)+1)], np.sqrt(train_score), label ="train")
    plt.plot([i for i in range(1, len(X_train)+1)], np.sqrt(test_score), label = "test")
    plt.legend
    plt.title("Learning Curve")
    plt.axis([0, len(X_train)+1, 0, 4])
    plt.show()

