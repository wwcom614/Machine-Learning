#coding:utf-8

import math

class UserBasedCF:
    def __init__(self,trainDataFile=None,testDataFile=None,splitor='\t'):
        # 读取训练数据文件，放入用户-物品评分矩阵train[userId][itemId]=score
        if trainDataFile!=None:
            self.train=self._loadData(trainDataFile, splitor)
        # 读取测试数据文件，放入用户-物品评分矩阵test[userId][itemId]=score
        if testDataFile!=None:
            self.test=self._loadData(testDataFile, splitor)
        self._userSimiMatrix= self.calUserSimilarity()

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

    # 计算输出 用户u-用户v 相似度矩阵
    def calUserSimilarity(self):
        # 因不是所有的物品都有人买，为降低计算量
        # 构建物品-不同购买用户列表的字典item_diffUsers：key是itemId，value是购买该item的不同用户set集合
        item_diffUsers=dict()
        for u,items in self.train.items():
            for i in items.keys():
                item_diffUsers.setdefault(i,set())
                item_diffUsers[i].add(u)
        # print("【item_diffUsers】：",item_diffUsers)
        # {'1605': {'782', '463', '901'}}

        # 每个用户-购买物品数量的字典
        user_itemsCount=dict()

        # 用户u-用户v共现矩阵
        co_useru_userv=dict()
        #因不是所有的物品都有人买，为降低计算量，不是遍历整个数据集，而是遍历物品-不同购买用户列表的字典item_diffUsers
        for item,users in item_diffUsers.items():
            for u in users:
                # 遍历item_diffUsers，按每个用户统计，累计每个用户-购买物品数量。
                # 构建每个用户-购买物品数量的字典user_itemsCount
                user_itemsCount.setdefault(u,0)
                user_itemsCount[u]+=1

                # 构建用户u-用户v共现矩阵co_useru_userv
                for v in users:
                    # 如果是本人，不打标记，直接跳过
                    if u==v:
                        continue
                    co_useru_userv.setdefault(u,{})
                    co_useru_userv[u].setdefault(v,0)
                    # 用户u和用户v对某itemId都有购买，计数+1
                    co_useru_userv[u][v]+=1
                # print("【co_useru_userv】：",co_useru_userv)
                #【co_useru_userv】：{450': {'903': 0.17054278655205474, '160': 0.17054278655205474}}

        # 用户u-用户v 相似度矩阵 {u:{v:相似度, ...}, ...}
        userSimiMatrix=dict()
        for u,related_users in co_useru_userv.items():
            userSimiMatrix.setdefault(u,{})
            for v,cuv in related_users.items():
                userSimiMatrix[u][v]=cuv/math.sqrt(user_itemsCount[u]*user_itemsCount[v])
        return userSimiMatrix

    # 给用户userU推荐topN个最相似的物品
    # 超参数userU_simiUsersTopN：用户u-用户v 相似度矩阵，取出与用户userU最相关的前userU_simiUsersTopN个用户
    def recommend(self,userU,userU_simiUsersTopN,topN=10):
        recommendItems=dict()
        # 获取用户userU已经购买过的物品和评分
        userU_items=self.train[userU]

        # 遍历用户u-用户v 相似度矩阵，取出与用户userU最相关的前userU_simiUsersTopN个用户
        for userV,simiUV in sorted(self._userSimiMatrix[userU].items(),key=lambda x :x[1],reverse=True)[0:userU_simiUsersTopN]:
            # 训练集中获取与用户userU购买行为最相似的用户userV购买的物品和评分
            for userV_item,userV_score in self.train[userV].items():
                # 如果与用户userU最相关的userV购买的物品，用户userU之前已经购买过了，直接跳过，不推荐该物品
                if userV_item in userU_items:
                    continue
                # 考虑相似度最高的物品userV_item的评分影响，相似度=simiUV * userV_score加权
                recommendItems.setdefault(userV_item, 0)
                recommendItems[userV_item]+=simiUV*float(userV_score)
        # 取simiUV * userV_score最高的topN个物品，作为最终推荐结果
        return dict(sorted(recommendItems.items(),key = lambda x :x[1],reverse = True)[0:topN])

    # 推荐效果评估--召回率Recall和精准率Precision
    def recallAndPrecision(self,userU_simiUsersTopN,topN=10):
        hit=0
        recall=0
        precision=0
        for user in self.train.keys():
            # 测试集中，真被用户购买的的物品列表testUserItems
            testUserItems=self.test.get(user,{})
            # 推荐的物品列表recommendItems
            recommendItems=self.recommend(user,userU_simiUsersTopN,topN)
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
    def coverage(self,userU_simiUsersTopN,topN=10):
        recommendItemSet=set()
        allItemSet=set()
        for user in self.train.keys():
            for item in self.train[user].keys():
                # 遍历所有用户购买的所有物品，放入set排重
                allItemSet.add(item)
            # 遍历所有用户推荐的所有物品，放入set排重
            for item,_ in self.recommend(user,userU_simiUsersTopN,topN).items():
                recommendItemSet.add(item)
        # 推荐的排重物品allItemSet数量 / 总排重物品recommendItemSet数量
        return len(recommendItemSet)/(len(allItemSet)*1.0)


    # 推荐效果评估--流行度，基尼系数，防马太效应
    def popularity(self,userU_simiUsersTopN,topN=10):
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
            for item, _ in self.recommend(user,userU_simiUsersTopN,topN).items():
                ret+=math.log(1+item_count[item])
                recommendItem_count+=1
        return ret/(recommendItem_count*1.0)


if __name__=="__main__":
    trainDataFile = r'D:\ideaworkspace\MachineLearning\DataSet\ml-100k\u3.base'
    testDataFile = r'D:\ideaworkspace\MachineLearning\DataSet\ml-100k\u3.test'
    ucf=UserBasedCF(trainDataFile,testDataFile,splitor='\t')
    print(ucf.recommend(userU='3', userU_simiUsersTopN=15))

    print("%2s%15s%15s%15s%15s" % ('userU_item_simiItemsTopN',"precision",'recall','coverage','popularity'))
    for userU_simiUsersTopN in [5,10,20,40,80,160]:
        recall,precision = ucf.recallAndPrecision(userU_simiUsersTopN = k)
        coverage = ucf.coverage(userU_simiUsersTopN = k)
        popularity = ucf.popularity(userU_simiUsersTopN = k)
        print("%2d%14.2f%%%14.2f%%%14.2f%%%15.2f" % (userU_simiUsersTopN,precision * 100,recall * 100,coverage * 100,popularity))