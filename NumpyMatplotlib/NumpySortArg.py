import numpy as np


s = np.random.normal(loc=0, scale=1, size=1000000)

#最大最小值的索引
print(np.argmax(s))
print(np.argmin(s))

a = np.arange(16)
#注意，此处打乱的是源a
np.random.shuffle(a)
print(a)

#注意，此处返回值排序，但源a并没有被排序
print(np.sort(a))

print(a)

#按值大小返回对应索引值
print(np.argsort(a))

#按排名第几小(可以为0，但不能>=数组维度大小)的值进行分组，左侧是小的，右侧是大的，两边内部并不排序
print(np.partition(a,3))

#注意，如果想直接排序a，使用如下方法
a.sort()
print(a)

np.random.seed(2)
X = np.random.randint(15,size=(3,5))
print(X)
'''
[[ 8 13  8  6 11]
 [ 2 11  8  7  2]
 [ 1 11  5 10  4]]
'''

#沿着行，在每行里按列数值排序
print(np.sort(X,axis=1))
'''
[[ 6  8  8 11 13]
 [ 2  2  7  8 11]
 [ 1  4  5 10 11]]
'''

#按排名第几小(可以为0，但不能>=数组维度大小)的值进行分组，左侧是小的，右侧是大的，两边内部并不排序
print(np.partition(X,2, axis=1))
'''
[[ 6  8  8 13 11]
 [ 2  2  7  8 11]
 [ 1  4  5 10 11]]
'''

#沿着列，在每列里按行数值排序
print(np.sort(X,axis=0))
'''
[[ 1 11  5  6  2]
 [ 2 11  8  7  4]
 [ 8 13  8 10 11]]
'''


