# coding:utf-8

# Apriori算法计算频繁项集的过程：
# (1)由数据集生成候选项集C1（1表示每个候选项仅有一个数据项）；再由C1通过支持度过滤，生成频繁项集L1（1表示每个频繁项仅有一个数据项）。
# (2)将L1的数据项两两拼接成C2。
# (3)从候选项集C2开始，通过支持度过滤生成L2。L2根据Apriori原理拼接成候选项集C3；C3通过支持度过滤生成L3……直到Lk中仅有一个或没有数据项为止。
# Apriori算法不适用于非重复项集数元素较多的案例，建议分析的商品种类10左右

# 读取数据集
def loadData(data):
    # 加载数据集data
    if data == None:
        print("Load dataSet error!")
        return

    # 将加载数据集data的每笔交易内按物品排重，录入dataSet
    dataSet = list(map(set, data))
    print("【dataSet】：", dataSet)
    # 【dataSet】： [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
    return dataSet


# 构建候选项集C1。使用frozenset标识C1为不可变集合
def _createC1(dataSet):
    C1 = []
    # 遍历数据集的每笔交易记录
    for transaction in data:
        # 遍历每笔交易记录的每个物品
        for item in transaction:
            # 每个物品item排重录入C1
            if not [item] in C1:
                C1.append([item])
    # C1中物品顺序排序
    C1.sort()
    # 使用frozenset标识C1为不可变集合
    C1 = list(map(frozenset, C1))
    return C1


# 候选项集Ck中，提取支持度>minSupport的项们，录入频繁项集Lk
# k的含义是每笔交易含k个物品
# 入参dataSet：物品排重数据集
# 入参Ck：候选项集
# 入参minSupport：超参数，最小支持度阈值
# 候选项集Ck由上一层(第k-1层)的频繁项集Lk-1组合得到
# 使用设置的超参数最小支持度minSupport对候选集Ck过滤
# 返回值：本层(第k层)的频繁项集Lk，以及每项的支持度
def _calFrequentItemsSetLk(dataSet, Ck, minSupport):
    # 每次遍历前，对数据集做过滤，只保留>=候选项集每笔交易中物品数量的交易记录，减少数据量和遍历次数，节约计算资源，提高计算性能
    print("【len(Ck[0])】：", len(Ck[0]))
    fileredDataSet = [item for item in dataSet if len(item) >= len(Ck[0])]

    # 定义字典candidate_count_dict，key是候选项，value是该候选项出现在数据集交易中的笔数
    candidate_count_dict = {}

    # 遍历过滤后的数据集的每笔交易
    for transaction in fileredDataSet:
        # 遍历候选项集Ck的每个候选项
        for candidate in Ck:
            # 如果候选项candidate是某笔交易candidate的子集
            if candidate.issubset(transaction):
                # 如果之前没有该记录，初始化计数1
                if not candidate in candidate_count_dict:
                    candidate_count_dict[candidate] = 1
                # 如果之前有该记录，计数累计+1
                else:
                    candidate_count_dict[candidate] += 1

    # 定义本层(第k层)的频繁项集Lk
    Lk = []
    # 定义字典candidate_support_dict，key是候选项，value是该候选项的支持度
    candidate_support_dict = {}
    # 遍历上面计算好的字典candidate_count_dict
    for key in candidate_count_dict:
        # 某候选项的支持度 = 该候选项出现在数据集交易中的笔数 / 数据集的总交易笔数
        support = candidate_count_dict[key] / len(dataSet) * 1.0
        # 只保留支持度>阈值的记录，从列表头部录入，形成频繁项集Lk
        if support >= minSupport:
            Lk.insert(0, key)
        # 顺便记录每次计算的support，形成项集支持度矩阵
        candidate_support_dict[key] = support
    return Lk, candidate_support_dict


# 从频繁项集Lk-1中，提取两两交集为k项的项集，合并后排重录入下一级候选项集Ck
# 入参Lk_1：频繁项集Lk-1，
# 入参k：本次计算k项的候选项集
def _calCandidateItemsSetCk(Lk_1, k):
    # k是从2开始的，也就是最开始传进来的是频繁项集L1，要计算候选项集C2
    Ck = []
    # 前一项与后面所有项逐个比较
    for i in range(0, len(Lk_1) - 1):
        L1 = Lk_1[i]
        for j in range(i + 1, len(Lk_1)):
            L2 = Lk_1[j]
            # print("【L1 & L2】:", L1 & L2)
            # print("【len(L1 & L2)】:", len(L1 & L2))
            # 频繁项集Lk-1的前后2个项集都是有k-1个元素的，
            # 两者相与如果有k-2个元素相同，只有1个元素不同，那么相或组合时才能组合成k个元素的候选项集Ck
            if len(L1 & L2) == k - 2:
                L1_2 = L1 | L2
                if L1_2 not in Ck:
                    Ck.append(L1_2)
    return Ck


# apriori算法-迭代寻找交易含k个物品的最大频繁项集Lk：C1->L1->C2->L2->C3->L3->....->Ck->Lk
def aprioriScanFrequentItemsSetLmax(dataSet, minSupport=0.5):
    print("===========apriori ScanFrequentItemsSetLmax start!===============")
    # 调用_createC1构建候选项集C1
    C1 = _createC1(dataSet)
    print("【候选项集C1】：", C1)
    # 【候选项集C1】： [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]

    # 调用_calFrequentItemsSetLk，计算出频繁项集L1和候选项-支持度矩阵candidate_support_dict。
    L1, candidate_support_dict = _calFrequentItemsSetLk(dataSet, C1, minSupport)
    print("【L1】：", L1)
    print("【candidate_support_dict】：", candidate_support_dict)

    # 将频繁项集压入一个列表，记录中间计算的每个Lk频繁项集的结果，并方便后面的迭代计算。
    L = [L1]
    # 从候选项集C2开始，通过支持度过滤生成L2。L2根据Apriori原理拼接成候选项集C3；C3通过支持度过滤生成L3……直到Lk中仅有一个或没有数据项为止。
    k = 2
    # 迭代计算的终止条件是找不到支持度>minSupport的最长频繁项集Lk
    while (len(L[k - 2]) > 0):
        Lk_1 = L[k - 2]
        Ck = _calCandidateItemsSetCk(Lk_1, k)
        print("【C" + str(k) + "】：", Ck)
        if (len(Ck) > 0):
            Lk, Ck_support_dict = _calFrequentItemsSetLk(dataSet, Ck, minSupport)
            # 注意：此处要记录k频繁项集的计算结果，然后update添加到candidate_support_dict中，直接赋值candidate_support_dict会被清空覆盖
            candidate_support_dict.update(Ck_support_dict)
            print("【L" + str(k) + "】：", Lk)
            L.append(Lk)
            k += 1
        else:
            break

    return L, candidate_support_dict


# 基于入参，计算输出满足最小置信度的推荐物品列表，当preItems_recommedItems只有2个时，直接计算即可得到推荐物品列表
# 入参preItems_recommedItems：某个频繁项(一组物品)，含义是前置物品+推荐物品的集合
# 入参probRecommedItems：某个频繁项freqSet中的每个物品集合，含义是可能被推荐待分析物品集合
# 入参supportData：前面计算好的项集支持度矩阵
# 入参recommendTuples：(前置物品,推荐物品,置信度)的列表
# 入参minConfidence：预制的超参数最小置信度
# 返回值recommendItems：推荐物品列表
def _calConfidence(preItems_recommedItems, probRecommedItems, supportData, recommendTuples, minConfidence=0.7):
    # 定义recommendItems，用于录入分析后的推荐物品
    recommendItems = []
    # 遍历推荐物品集合中的每个物品
    for recommedItem in probRecommedItems:
        # 推荐的前置物品
        preItems = preItems_recommedItems - recommedItem
        # 该物品的置信度= 前置物品+推荐物品的支持度 / 前置物品的支持度
        confidence = supportData[preItems_recommedItems] / supportData[preItems]
        # 过滤，只保留置信度高于预制置信度的数据
        if confidence >= minConfidence:
            print(preItems, '-->', recommedItem, 'confidence:', confidence)
            # 元组中的三个元素：前置物品集、推荐物品、置信度
            recommendTuples.append((preItems, recommedItem, confidence))
            recommendItems.append(recommedItem)
    # 返回推荐物品列表
    return recommendItems


# 当preItems_recommedItems>2个时，评估频繁项集中元素超过2个的项集进行合并
# https://www.cnblogs.com/bigmonkey/p/7449761.html
# 入参preItems_recommedItems：某个频繁项(一组物品)，含义是前置物品+推荐物品集合
# 入参probRecommedItems：某个频繁项freqSet中的每个物品集合，含义是可能被推荐待分析物品集合
# 入参supportData：前面计算好的项集支持度矩阵
# 入参recommendTuples：(前置物品,推荐物品,置信度)的列表
#
def _rulesFromConseq(preItems_recommedItems, probRecommedItems, supportData, recommendTuples, minConfidence=0.7):
    # 定义recommendItems，用于录入分析后的推荐物品
    recommendItems = []

    # 前置物品+推荐物品集合preItems_recommedItems 要比 一个probRecommedItem元素 至少多2个，
    # 多1个直接使用_calConfidence，即可计算出recommendTuples，对应已知多个物品，推荐1个物品的场景
    # probRecommedItems经过_calCandidateItemsSetCk计算后，衍生出多物品组合，对应已知多个物品，推荐多个物品的场景
    if len(preItems_recommedItems) > len(probRecommedItems[0]) + 1:
        # m个物品项集 组合成 m+1个物品项集
        recommendItems = _calCandidateItemsSetCk(probRecommedItems, len(probRecommedItems[0]) + 1)
        # 看看新构建的m+1个物品项集，在freqSet基础上，置信度如何
        recommendItems = _calConfidence(preItems_recommedItems, recommendItems, supportData, recommendTuples, minConfidence)

        # 如果不止一条规则满足要求，进一步递归合并
        if len(recommendItems) > 1:
            _rulesFromConseq(preItems_recommedItems, recommendItems, supportData, recommendTuples, minConfidence)


# 根据之前计算出的频繁项集L和预置超参数最小置信度生成推荐组合列表
def aprioriGenerateRecommendTuples(L, supportData, minSupport=0.7):
    # 推荐组合列表
    recommendTuples = []
    # 对于寻找关联规则来说，频繁1项集L1没有用处，因为L1中的每个集合仅有一个数据项，至少有两个数据项才能生成A→B这样的关联规则
    # 所以从频繁项集L2开始逐个遍历
    for k in range(1, len(L)):
        # 遍历每个频繁项集Lk中的每个频繁项preItems_recommedItems(一组物品)
        for preItems_recommedItems in L[k]:
            # recommedItems是每个preItems_recommedItems中的每个物品集合，每个都有可能是推荐物品
            recommedItems = [frozenset([item]) for item in preItems_recommedItems]

            # 总共2个物品的频繁项集L2中，寻找置信度超过minConfidence的A→B的组合
            if k == 1:
                _calConfidence(preItems_recommedItems, recommedItems, supportData, recommendTuples, minSupport)
            # 超过2个物品的频繁项集Lk
            else:
                _rulesFromConseq(preItems_recommedItems, recommedItems, supportData, recommendTuples, minSupport)

    return recommendTuples


if __name__ == "__main__":
    data = [[1, 3, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]]

    dataSet = loadData(data)
    # 【dataSet】： [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
    L, candidate_support_dict = aprioriScanFrequentItemsSetLmax(dataSet=dataSet, minSupport=0.3)
    print("【L】：", L)
    print("【supportData】：", candidate_support_dict)
    '''
    ===========apriori ScanFrequentItemsSetLmax start!===============
【候选项集C1】： [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
【len(Ck[0])】： 1
【L1】： [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
【candidate_support_dict】： {frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75}
【C2】： [frozenset({2, 5}), frozenset({3, 5}), frozenset({1, 5}), frozenset({2, 3}), frozenset({1, 2}), frozenset({1, 3})]
【len(Ck[0])】： 2
【L2】： [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]
【C3】： [frozenset({2, 3, 5}), frozenset({1, 2, 3}), frozenset({1, 3, 5})]
【len(Ck[0])】： 3
【L3】： [frozenset({2, 3, 5})]
【C4】： []
【L】： [[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})], [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})], [frozenset({2, 3, 5})]]
【supportData】： {frozenset({1}): 0.5, frozenset({3}): 0.75, frozenset({4}): 0.25, frozenset({2}): 0.75, frozenset({5}): 0.75, frozenset({1, 3}): 0.5, frozenset({2, 5}): 0.75, frozenset({3, 5}): 0.5, frozenset({2, 3}): 0.5, frozenset({1, 5}): 0.25, frozenset({1, 2}): 0.25, frozenset({2, 3, 5}): 0.5, frozenset({1, 2, 3}): 0.25, frozenset({1, 3, 5}): 0.25}
    '''

    recommendTuples = aprioriGenerateRecommendTuples(L=L, supportData=candidate_support_dict, minSupport=0.6)
    print("【recommendTuples】：", recommendTuples)
