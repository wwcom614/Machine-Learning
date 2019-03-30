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

    # 计算输出 物品i-物品j 相似度矩阵
    def calItemsSimilarity(self):
        # 构建每个物品-购买用户数量的字典item_usersCount
        item_usersCount = dict()

        # 构建物品i-物品j共现矩阵co_itemi_itemj
        co_itemi_itemj = dict()

        # 遍历训练数据集，获取到所有用户购买的items
        for user,item_score_pairs in self.train.items():
            # 再遍历所有items
            for i in item_score_pairs.keys():
                # 生成每个物品-购买用户数量的字典item_usersCount
                item_usersCount.setdefault(i,0)
                item_usersCount[i] += 1

                # 构建物品i-物品j共现矩阵
                co_itemi_itemj.setdefault(i,{})
                for j in item_score_pairs.keys():
                    # 物品i-物品j共现矩阵中，如果是自身，不打标记，直接跳过
                    if j==i:
                        continue
                    # 物品i 和 物品j 有重合购买用户，计数+1
                    co_itemi_itemj[i].setdefault(j,0)
                    co_itemi_itemj[i][j] += 1

        # 物品i-物品j 相似度矩阵 {i:{j:相似度, ...}, ...}
        itemSimiMatrix=dict()
        #遍历物品i-物品j共现矩阵
        for i, related_items in co_itemi_itemj.items():
            itemSimiMatrix.setdefault(i,{})
            for j, cij in related_items.items():
                # 物品i和物品j的余弦相似度矩阵。
                # 注：物品i的物品j余弦相似度 与 物品j的物品i余弦相似度 并不相等
                itemSimiMatrix[i][j] = cij / math.sqrt(item_usersCount[i] * item_usersCount[j])

        return itemSimiMatrix


    # 给用户userU推荐topN个最相似的物品
    # 超参数userU_item_simiItemsTopN：对用户userU购买的每种物品，最多取多少个最相似的物品
    def recommend(self,userU,userU_item_simiItemsTopN,topN=10):
        recommendItems=dict()
        # 获取用户userU已经购买过的物品和评分
        userU_items=self.train[userU]

        # 遍历用户userU已经购买过的物品
        for itemI, score in userU_items.items():
            # 遍历用户userU购买的每种物品i-物品j 相似度矩阵 {i:{j:相似度, ...}, ...},每次按相似度倒序排序，取相似度最高的前userU_item_simiItemsTopN组"物品i-物品j"，
            # 最多会取出用户userU购买过的  所有物品数*userU_item_simiItemsTopN  组数据
            for itemJ, simiIJ in sorted(self._itemSimiMatrix[itemI].items(), key=lambda x:x[1], reverse=True)[0:userU_item_simiItemsTopN]:
                # 如果相似度矩阵中的物品j是用户userU已经购买过的物品i，直接跳过，不推荐该物品
                if itemJ==itemI:
                    continue
                # 考虑相似度最高的物品itemJ的评分影响，相似度=simiIJ * score加权
                recommendItems.setdefault(itemJ, 0)
                recommendItems[itemJ] += simiIJ * float(score)
        # 取simiIJ * score最高的topN个物品，作为最终推荐结果
        return dict(sorted(recommendItems.items(),key = lambda x :x[1],reverse = True)[0:topN])


    # 推荐效果评估--召回率Recall和精准率Precision
    def recallAndPrecision(self,userU_item_simiItemsTopN,topN=10):
        hit=0
        recall=0
        precision=0
        for user in self.train.keys():
            # 测试集中，真被用户购买的的物品列表testUserItems
            testUserItems=self.test.get(user,{})
            # 推荐的物品列表recommendItems
            recommendItems=self.recommend(user,userU_item_simiItemsTopN,topN)
            for item ,_ in recommendItems.items():
                if item in testUserItems:
                    # 推荐的物品 且
                    hit+=1
            # 召回率的分母是测试集中所有用户真实购买物品数量
            recall+=len(testUserItems)
            # 精准率的分母是为所有用户预测的物品结果数量
            precision+=topN
        #print 'Recall:%s    hit:%s    allRatings:%s'%(hit/(recall*1.0),hit,precision)
        return (hit / (recall * 1.0),hit / (precision * 1.0))

    # 推荐效果评估--覆盖率
    def coverage(self,userU_item_simiItemsTopN,topN=10):
        recommendItemSet=set()
        allItemSet=set()
        for user in self.train.keys():
            for item in self.train[user].keys():
                # 遍历所有用户购买的所有物品，放入set排重
                allItemSet.add(item)
            # 遍历所有用户推荐的所有物品，放入set排重
            for item,_ in self.recommend(user,userU_item_simiItemsTopN,topN).items():
                recommendItemSet.add(item)
        # 推荐的排重物品allItemSet数量 / 总排重物品recommendItemSet数量
        return len(recommendItemSet)/(len(allItemSet)*1.0)


    # 推荐效果评估--流行度，基尼系数，防马太效应
    def popularity(self,userU_item_simiItemsTopN,topN=10):
        # 每个item及其被购买的次数
        item_count=dict()
        for user,items in self.train.items():
            for item in items.keys():
                if item not in item_count:
                    item_count[item]=1
                item_count[item]+=1
        ret=0
        # 所有用户被推荐的每个物品被购买的次数
        recommendItem_count=0
        for user in self.train.keys():
            for item, _ in self.recommend(user,userU_item_simiItemsTopN,topN).items():
                ret+=math.log(1+item_count[item])
                recommendItem_count+=1
        return ret/(recommendItem_count*1.0)


if __name__=="__main__":
    trainDataFile = r'D:\ideaworkspace\MachineLearning\DataSet\ml-100k\u3.base'
    testDataFile = r'D:\ideaworkspace\MachineLearning\DataSet\ml-100k\u3.test'
    icf=ItemBasedCF(trainDataFile=trainDataFile,testDataFile=testDataFile,splitor='\t')
    print(icf.recommend(userU='3', userU_item_simiItemsTopN=20))

    print("%2s%15s%15s%15s%15s" % ('userU_item_simiItemsTopN',"precision",'recall','coverage','popularity'))
    for userU_item_simiItemsTopN in [5,10,20,40,80,160]:
        recall,precision = icf.recallAndPrecision(userU_item_simiItemsTopN = userU_item_simiItemsTopN)
        coverage = icf.coverage(userU_item_simiItemsTopN = userU_item_simiItemsTopN)
        popularity = icf.popularity(userU_item_simiItemsTopN = userU_item_simiItemsTopN)
        print("%3d%14.2f%%%14.2f%%%14.2f%%%15.2f" % (userU_item_simiItemsTopN, precision * 100,recall * 100,coverage * 100, popularity))
