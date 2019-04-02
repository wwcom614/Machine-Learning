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


# 候选项集Ck，提取支持度>minSupport的项们，录入频繁项集Lk
# 入参：物品排重数据集dataSet、候选项集Ck、最小支持度阈值minSupport
# 候选项集Ck由上一层(第k-1层)的频繁项集Lk-1组合得到
# 使用设置的超参数最小支持度minSupport对候选集Ck过滤
# 返回值：本层(第k层)的频繁项集Lk，以及每项的支持度
def _calFrequentItemsSetLk(dataSet, Ck, minSupport):
    # 定义字典candidate_count_dict，key是候选项，value是该候选项出现在数据集交易中的笔数
    candidate_count_dict = {}

    # 计算前，对每次要遍历的数据集做好过滤，减少数据量，提高计算性能
    print("【len(Ck[0])】：", len(Ck[0]))
    fileredDataSet = [item for item in dataSet if len(item) >= len(Ck[0])]

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
        candidate_support_dict[key] = support
    return Lk, candidate_support_dict


# 入参：频繁项集Lk-1，本次计算k项的候选项集
# 连接转换为k项的候选项集Ck
def _calCandidateItemsSetCk(Lk_1, k):
    # k是从2开始的，也就是最开始传进来的是频繁项集L1，要计算候选项集C2
    Ck = []
    # 前一项与后面所有项逐个比较
    for i in range(0, len(Lk_1)-1):
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


# apriori
def apriori(dataSet, minSupport=0.5):
    print("===========apriori start!===============")
    # 构建候选项C1项集。使用frozenset标识_C1为不可变集合
    C1 = _createC1(dataSet)
    print("【1-项集C1】：", C1)
    # 【1-项集C1】： [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]

    # 基于数据集dataSet、候选项集C1和设置的超参数支持度minSupport，计算出频繁项集L1和候选项-支持度矩阵candidate_support_dict。
    L1, candidate_support_dict = _calFrequentItemsSetLk(dataSet, C1, minSupport)
    print("【L1】：", L1)
    print("【candidate_support_dict】：", candidate_support_dict)

    # 将频繁项集压入一个列表，便于后面的迭代计算
    L = [L1]
    # 下一步要开始迭代计算k>=2的频繁项集
    k = 2
    # 迭代计算的终止条件是找不到支持度>minSupport的最长频繁项集Lk
    while (len(L[k - 2]) > 0):
        Lk_1 = L[k - 2]
        Ck = _calCandidateItemsSetCk(Lk_1, k)
        print("【C" + str(k) + "】：", Ck)
        if (len(Ck) > 0):
            Lk, supportK = _calFrequentItemsSetLk(dataSet, Ck, minSupport)
            print("【L" + str(k) + "】：", Lk)
            L.append(Lk)
            k += 1
        else:
            break

    return L, supportK


if __name__ == "__main__":
    data = [[1, 3, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]]

    dataSet = loadData(data)
    L, supportData = apriori(dataSet=dataSet, minSupport=0.3)
    print("【L】：", L)
    print("【supportData】：", supportData)
