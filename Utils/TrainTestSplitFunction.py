import numpy as np

def train_test_split(X, y, test_ratio = 0.2 ,seed = 0):
    assert X.shape[0] == y.shape[0], "the size of X must be the same as y !"
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be more than 0 and less than 1 !"

    #seed是为了调试时固定random随机index
    if seed:
        np.random.seed(seed)

    #将样本分为训练集和测试集。但不能直接按段划分，因为该数据集前A个是第1种花，中间B个是第2种花，最后C个是第3种花
    #所以必须整体先随机乱序，再划分
    #np.random.permutation是将X行数(样本数150)随机生成index数组
    shuffle_index = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)

    test_indexes = shuffle_index[:test_size]
    train_indexes = shuffle_index[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test