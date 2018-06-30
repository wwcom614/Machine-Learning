import numpy as np


#concatenate用途，新增样本的融入

x = np.array([1,2,3])
y = np.array([4,5,6])
z = np.array([7,8,9])
#print(np.concatenate([x,y,z]))
# [1 2 3 4 5 6 7 8 9]

A = np.array([[1,2,3], [4,5,6]])
B = np.array([[7,8,9], [10,11,12]])

#axis=0,延第一个维度拼接，行拼接
#print(np.concatenate([A,B],axis=0))
'''
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
'''

#axis=1,延第二个维度拼接，列拼接
#print(np.concatenate([A,B],axis=1))
'''
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
'''

#列相同，纵向加行
#concatenate的数据必须要要相同维度，所以z向量要reshape
#all the input arrays must have same number of dimensions
#print(z.reshape(1,-1))
#print(np.concatenate([A,z.reshape(1,-1)]))

#兼容性更好的vstack，列相同，纵向加行
#print(np.vstack([A,z]))
'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''

D = np.array([[11,12],[13,14]])

#行相同，横向加列，如下两种方式都OK
#print(np.concatenate([A,D],axis=1))
#print(np.hstack([A,D]))
'''
[[ 1  2  3 11 12]
 [ 4  5  6 13 14]]
'''
##########################################

#split 分割操作，应用场景：将样本分为特征和标识 X,y = np.hsplit(data, [-1])
#a = np.arange(10)
#print(a)
#[0 1 2 3 4 5 6 7 8 9]

#x1, x2, x3 = np.split(a, [3,7])
#print(x1)
#[0 1 2]

#print(x2)
#[3 4 5 6]

#print(x3)
#[7 8 9]


A = np.arange(16).reshape(4,4)
print(A)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
'''

#按行拆分，两种方式
#up, down = np.split(A, [2], axis = 0)
#up, down = np.vsplit(A, [2])
#print(up)
'''
[[0 1 2 3]
 [4 5 6 7]]
'''

#print(down)
'''
[[ 8  9 10 11]
 [12 13 14 15]]
'''

#按列拆分，两种方式
#left, right = np.split(A, [-1], axis = 1)
left, right = np.hsplit(A, [-1])
print(left)
'''
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]
 [12 13 14]]
'''

print(right)
'''
[[ 3]
 [ 7]
 [11]
 [15]]
'''

#矩阵变向量
y = right[:, 0]
print(y)