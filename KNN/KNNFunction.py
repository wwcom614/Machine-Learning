import numpy as np
from math import sqrt
from collections import Counter

def KNN_Fuction(K, X_train, y_train, predictx):
    assert 1 <= K <= X_train.shape[0], "K must be more than 1 and less than the size of X_train!"
    assert X_train.shape[0] == y_train.shape[0], "The size of X_train must be the same as y_train!"
    assert X_train.shape[1] == predictx.shape[0], "The features of  X_train must be the same as predictx"

    distances = [sqrt(np.sum((x_train - predictx) ** 2)) for x_train in X_train]
    nearestIndex = np.argsort(distances)


    topK_y = y_train[nearestIndex[:K]]

    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
