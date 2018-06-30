import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

print(iris.keys())
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

#print(iris.DESCR)
#有150个样本，4个特征，3个标签class分类
'''
Data Set Characteristics:
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
'''


print(iris.data)
'''
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 ......
'''

print(iris.data.shape)
#(150, 4)

print(iris.feature_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print(iris.target)
#对应data150个样本的label 0,1,2
print(iris.target_names)
#label 0,1,2 分别表示 ['setosa' 'versicolor' 'virginica']

############################
#数据演练
#先看看样本前2列

#sepal length,取样本数据第1列
sl = iris.data[:, 0]

#sepal width,取样本数据第2列
sw = iris.data[:, 1]

#结果
label = iris.target

plt.scatter(sl[label==0], sw[label==0], color="red", label="Iris-Setosa", marker="o")
plt.scatter(sl[label==1], sw[label==1], color="blue", label="Iris-Versicolour", marker="+")
plt.scatter(sl[label==2], sw[label==2], color="green", label="Iris-Virginica", marker="x")

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("The relations of iris and sepal length and sepal width scatter")
plt.legend() #加上label图例
plt.show()

#################################
#再看看样本后2列

#sepal length,取样本数据第3列
sl = iris.data[:, 2]

#sepal width,取样本数据第4列
sw = iris.data[:, 3]

#结果
label = iris.target

plt.scatter(sl[label==0], sw[label==0], color="red", label="Iris-Setosa", marker="o")
plt.scatter(sl[label==1], sw[label==1], color="blue", label="Iris-Versicolour", marker="+")
plt.scatter(sl[label==2], sw[label==2], color="green", label="Iris-Virginica", marker="x")

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("The relations of iris and petal length and petal width scatter")
plt.legend() #加上label图例
plt.show()











