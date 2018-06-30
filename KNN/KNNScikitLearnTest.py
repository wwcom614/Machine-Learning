from sklearn.neighbors import KNeighborsClassifier
import numpy as np

##############################################
# 构造样本数据
raw_data_X = [[3.393533211,2.331273381],
              [3.110073483,1.781539638],
              [1.343808831,3.368360954],
              [3.582294042,4.679179110],
              [2.280362439,2.866990263],
              [7.423436942,4.696522875],
              [5.745051997,3.533989803],
              [9.172168622,2.511101045],
              [7.792786481,3.424088941],
              [7.939820817,0.791637231]]

raw_data_y = [0,0,0,0,0,1,1,1,1,1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 构造预测记录。需要将数组通过reshape(1, -1)转换为矩阵，再传给KNN_classfier.predict计算
predictx = np.array([8.093607318,3.365731514]).reshape(1, -1)


##############################################
# KNN算法 用scikit-learn的封装函数实现
KNN_classfier = KNeighborsClassifier(n_neighbors=6)

KNN_classfier.fit(X_train, y_train)

#注意：scikit-learn的predict参数只接受矩阵
print(KNN_classfier.predict(predictx)[0])
# 1


