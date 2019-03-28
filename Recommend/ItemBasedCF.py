#coding:utf-8

import math

class ItemBasedCF:

    def __init__(self,trainDataFile=None,testDataFile=None,splitor='\t'):
        # 读取训练数据文件，放入用户-物品评分矩阵train[userId][itemId]=score
        if trainDataFile!=None:
            self.train=self._loadData(trainDataFile, splitor)
        # 读取测试数据文件，放入用户-物品评分矩阵test[userId][itemId]=score
        if testDataFile!=None:
            self.test=self._loadData(testDataFile, splitor)
        self._itemSimiMatrix= self.calItemsSimilarity()

    # 读取数据文件，形成user-item的score评分表
    # 数据文件记录格式：  userId    itemId    score    timestamp，
    def _loadData(self,dataFile,splitor='\t'):
        data={}
        for line in open(dataFile):
            # 获取用户、物品、评分字段。计算相似度，不需要时间戳字段，_丢弃
            userId, itemId, score, _ = line.split(splitor)
            # 用户-物品评分矩阵
            data.setdefault(userId,{})
            data[userId][itemId]=score
        return data

    # 计算输出 物品i-物品j 相似度矩阵、topN热点商品
    def calItemsSimilarity(self):
        # 物品被多少个用户购买过
        item_users = dict()

        # 物品-物品共现矩阵
        Co_item_item = dict()

        #遍历训练数据集，获取有用户行为的item
        for user,item_scores in self.train.items():
            # 遍历每个item
            for i in item_scores.keys():
                # 每个物品-购买用户数
                item_users.setdefault(i,0)
                item_users[i] += 1

                # 物品i-物品j共现矩阵
                Co_item_item.setdefault(i,{})
                for j in item_scores.keys():
                    # 物品i-物品j共现矩阵中，如果是自身，不打标记，直接跳过
                    if j==i:
                        continue
                    # 物品i 和 物品j 有重合购买用户，计数+1
                    Co_item_item[i].setdefault(j,0)
                    Co_item_item[i][j] += 1

        # 物品i-物品j 相似度矩阵 {i:{j:相似度, ...}, ...}
        itemSimiMatrix=dict()
        #遍历物品i-物品j共现矩阵
        for i, related_items in Co_item_item.items():
            itemSimiMatrix.setdefault(i,{})
            for j, cij in related_items.items():
                # 物品i和物品j的余弦相似度矩阵。
                # 注：物品i的物品j余弦相似度 与 物品j的物品i余弦相似度 并不相等
                itemSimiMatrix[i][j] = cij / math.sqrt(item_users[i] * item_users[j])

        return itemSimiMatrix


    # 给用户userU推荐topN个最相似的物品
    #peersCount是指的，对用户userU购买的每种物品，最多取多少个最相似的物品
    def recommend(self,userU,peersCount,topN=10):
        recommendItems=dict()
        # 获取用户userU已经购买过的物品和评分
        interacted_items=self.train[userU]

        # 遍历用户userU已经购买过的物品
        for i,score in interacted_items.items():
            # 遍历物品i-物品j 相似度矩阵 {i:{j:相似度, ...}, ...},每次按相似度倒序排序，取相似度最高的前peersCount组"物品i-物品j"，
            # 最多会取出用户userU购买过的物品数*peersCount组数据
            for j, wj in sorted(self._itemSimiMatrix[i].items(), key=lambda x:x[1], reverse=True)[0:peersCount]:
                # 如果相似度矩阵中的物品j是用户userU已经购买过的物品i，直接跳过，不再推荐
                if j==i:
                    continue
                # 考虑相似度最高的物品j的评分影响，wj * score加权
                recommendItems.setdefault(j, 0)
                recommendItems[j] += wj * float(score)
        # 取wj * score最高的topN个物品j，作为最终推荐结果
        return dict(sorted(recommendItems.items(),key = lambda x :x[1],reverse = True)[0:topN])


    # 推荐效果评估--召回率Recall和精准率Precision
    def recallAndPrecision(self,peersCount,topN=10):
        hit=0
        recall=0
        precision=0
        for user in self.train.keys():
            # 获取某用户的商品
            testUserItems=self.test.get(user,{})
            recommendItems=self.recommend(user,peersCount,topN)
            for item ,_ in recommendItems.items():
                if item in testUserItems:
                    hit+=1
            # 召回率的分母是测试集中用户真实购买数量
            recall+=len(testUserItems)
            # 精准率的分母是预测结果数量
            precision+=topN
        #print 'Recall:%s    hit:%s    allRatings:%s'%(hit/(recall*1.0),hit,precision)
        return (hit / (recall * 1.0),hit / (precision * 1.0))

    # 推荐效果评估--覆盖率
    def coverage(self,peersCount,topN=10):
        recommendItemSet=set()
        allItemSet=set()
        for user in self.train.keys():
            for item in self.train[user].keys():
                # 遍历所有用户购买的所有物品，放入set排重
                allItemSet.add(item)
            # 遍历所有用户推荐的所有物品，放入set排重
            for item,_ in self.recommend(user,peersCount,topN).items():
                recommendItemSet.add(item)
        # 推荐的物品数量 占 总物品数量 比例
        return len(recommendItemSet)/(len(allItemSet)*1.0)


    # 推荐效果评估--流行度，基尼系数，防马太效应
    def popularity(self,peersCount,topN=10):
        # 每个item被购买的次数
        item_count=dict()
        for user,items in self.train.items():
            for item in items.keys():
                if item not in item_count:
                    item_count[item]=1
                item_count[item]+=1
        ret=0
        # 被推荐的物品被购买的次数
        recommendItem_count=0
        for user in self.train.keys():
            for item, _ in self.recommend(user,peersCount,topN).items():
                ret+=math.log(1+item_count[item])
                recommendItem_count+=1
        return ret/(recommendItem_count*1.0)


if __name__=="__main__":
    trainDataFile = r'D:\ideaworkspace\MachineLearning\DataSet\ml-100k\u3.base'
    testDataFile = r'D:\ideaworkspace\MachineLearning\DataSet\ml-100k\u3.test'
    icf=ItemBasedCF(trainDataFile=trainDataFile,testDataFile=testDataFile,splitor='\t')

    print("%2s%15s%15s%15s%15s" % ('peersCount',"precision",'recall','coverage','popularity'))
    for peersCount in [5,10,20,40,80,160]:
        recall,precision = icf.recallAndPrecision(peersCount = peersCount)
        coverage = icf.coverage(peersCount = peersCount)
        popularity = icf.popularity(peersCount = peersCount)
        print("%3d%14.2f%%%14.2f%%%14.2f%%%15.2f" % (peersCount, precision * 100,recall * 100,coverage * 100, popularity))
