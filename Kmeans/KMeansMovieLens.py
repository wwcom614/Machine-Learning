# -*- coding:utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.metrics import adjusted_rand_score
import datetime

# 封装读取数据文件，提取特征和标记
# 数据记录格式：userId    itemId    score    timestamp
# 是某用户userId，在时间点timestamp(精确到秒)，对某电影itemId的评分score。
def loadData(dataFile,splitor='\t'):
    X_data=[]
    y_data=[]
    for line in open(dataFile):
        fields = line.strip().split("\t")
        # 时间特征精确到秒是不合适的，因为人的兴趣变化没那么快，所以转换时间精确到天
        dt = datetime.datetime.utcfromtimestamp(int(fields[3])).strftime("%Y-%m-%d")
        month = int(dt.split("-")[1])
        day = int(dt.split("-")[2])
        # 用户userId、电影itemId、发生的月份、日期，作为特征
        X_data.append([int(fields[0]), int(fields[1]), month, day])
        # 评分score作为标记
        y_data.append(int(fields[2]))
    return X_data, y_data

# 样本数据特征矩阵
X_sample = []
# 样本数据标记矩阵
y_sample = []
# 训练数据文件
trainDataFile = "D:/ideaworkspace/MachineLearning/DataSet/ml-100k/u1.base"
# 调用封装的loadData方法，得到样本数据特征矩阵和标记矩阵
X_sample, y_sample = loadData(dataFile=trainDataFile,splitor='\t')

# 测试数据特征矩阵
X_test = []
# 测试数据标记矩阵
y_test = []
# 测试数据文件
testDataFile = "D:/ideaworkspace/MachineLearning/DataSet/ml-100k/u1.test"
# 调用封装的loadData方法，得到测试数据特征矩阵和标记矩阵
X_test, y_test = loadData(dataFile=testDataFile,splitor='\t')

#########################################################################

# 使用不同的Kmeans算法训练拟合，5个中心点(因为评分score是1~5)，训练出模型model
kmeans_model = KMeans(n_clusters=5, random_state=1).fit(X_sample, y_sample)

minibatchkmeans_model = MiniBatchKMeans(n_clusters=5, batch_size=20000, random_state=1).fit(X_sample, y_sample)

# 机器太挫跑不动-_-!!!
#birch_model = Birch(n_clusters=5, branching_factor=5, threshold=0.05, compute_labels=True).fit(X_sample, y_sample)

#########################################################################

# 不同Kmeans算法模型的效果评估
y_kmeans_predict = kmeans_model.predict(X_test)
print("【kmeans】：", adjusted_rand_score(y_test, y_kmeans_predict) * 10000)
# 【kmeans】： 136.0111476591705

y_minibatchkmeans_predict = minibatchkmeans_model.predict(X_test)
print("【minibatchkmeans】：", adjusted_rand_score(y_test, y_minibatchkmeans_predict) * 10000)
# 【minibatchkmeans】： 166.7603423161873

# 机器太挫跑不动-_-!!!
#y_birch_predict = birch_model.predict(X_test)
#print("【birch】：", adjusted_rand_score(y_test, y_birch_predict) * 10000)

