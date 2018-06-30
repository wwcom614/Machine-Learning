import numpy as np

#FancyIndex
x = np.arange(10,26)
print(x)
#[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]

print(x[3:9:2])
#[13 15 17]

#注意：中间有逗号
ind = [3,5,8]
print(x[ind])
#[13 15 18]

#分别取2次，组成新的二维矩阵
ind = np.array([[0,2], [1,3]])
print(x[ind])
'''
[[10 12]
 [11 13]]
'''

X = x.reshape(4,-1)
print(X)
'''
[[10 11 12 13]
 [14 15 16 17]
 [18 19 20 21]
 [22 23 24 25]]
'''

row = np.array([0, 1, 2])
col = np.array([1, 2, 3])
print(X[row,col])
#[11 16 21]

print(X[:,col])
'''
[[11 12 13]
 [15 16 17]
 [19 20 21]
 [23 24 25]]
'''

col = [True, False, True, True]
print(X[1:3, col])
'''
[[14 16 17]
 [18 20 21]]
'''

####################################
#Compare

print(X <= 13)
'''
[[ True  True  True  True]
 [False False False False]
 [False False False False]
 [False False False False]]
'''

#非运算符 ~
print(np.sum(~((X <= 16) & (X > 13))))
#13

print(np.count_nonzero((X <= 16) & (X > 13)))
#3

print(np.any(X == 13))
#True

#沿着列看，每行元素是否都 >=13
print(np.all(X >= 13,axis=1))
#[False  True  True  True]

#沿着行看，每列有多少个偶数
print(np.sum(X%2 == 0, axis=0))
#[4 0 4 0]

####################################
#Compare with FancyIndex
#第4列可以被3整除的 行记录
print(X[X[:,3] %3 == 0])
#[[18 19 20 21]]

'''
pandas库可以将数据抽象成dataframe，这样对数据的提取等操作能力更强大
但scikit-learn机器学习的数据输入对numpy的矩阵支持较好
所以一般会用pandas先处理，最后转成numpy的矩阵，传入scikit-learn
'''






