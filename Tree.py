from sklearn import tree
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
"""
这是一个奇怪的库:

它封装了一棵树,一个决策树,和一个用树解释决策树的决策树解释器

好吧其实我就是想写一个决策树解释器罢了,之前打数模的时候都不会解释决策树,好菜
"""


class Node:
    def __init__(self, info):
        """
        :param info: 节点包含的信息
        """
        self.info = info
        self.sons = []

    def add_sons(self, son_id):
        """
        :param son_id: 节点的儿子的ID的列表
        :return: none
        """
        self.sons.append(son_id)


class BinTree:  # 节点的坐标全部用数字表示,不要乱,同时我要说这是一颗二叉树
    def __init__(self, rootID=1):
        """
        :param rootID: 根节点ID,一般为1
        """
        self.rootID = rootID
        self.TreeList = []
        self.sequence = [Node("None")]
        self.p = 1

    def sonsOf(self, Id):
        """
        :param Id: 要查询的节点的ID
        :return: 返回Id节点的儿子的个数
        """
        try:
            return len(self.TreeList[Id].sons)
        except Exception as e:
            print("没有此节点")

    def addPoint(self, u, info):
        """
        :param u: 添加节点的ID
        :param info: 添加的节点的信息
        :return: none
        """
        while len(self.TreeList) < u + 10:
            self.TreeList.append(Node("EMPTY"))
        if self.TreeList[u].info != "EMPTY":
            return
        self.TreeList[u] = Node(info)

    def addEdge(self, fatherNode, v):  # 只连边不加点
        """
        :param fatherNode: 父节点ID
        :param v: 子节点ID
        :return: none
        """
        if len(self.TreeList) - 1 < max(fatherNode, v):
            print("点数{}超出了现有的点的个数{},加边失败".format(self.TreeList, max(fatherNode, v)))
            return
        self.TreeList[fatherNode].add_sons(v)

    def addRelationShip(self, fa, fa_info, son, son_info):
        """
        :param fa: 父节点的ID
        :param fa_info: 父节点的消息
        :param son: 子节点的ID
        :param son_info: 子节点的消息
        :return: none
        """
        self.addPoint(fa, fa_info)
        self.addPoint(son, son_info)
        self.addEdge(fa, son)

    mergeMessageForLeaves_data = []

    def dfs_for_merge(self, u, info: list):
        """
        :param u: dfs的开始节点
        :param info: 累积到此id的info
        :return: none
        注:信息缓存在mergeMessageForLeaves_data中
        """
        if self.sonsOf(u) == 0:
            self.mergeMessageForLeaves_data.append(info + [self.TreeList[u].info])
            return
        for sons in self.TreeList[u].sons:
            self.dfs_for_merge(sons, info + [self.TreeList[u].info])

    def mergeMessageForLeaves(self):
        """
        :return: 当前存放的树中每个子节点到父节点路径的信息总和
        """
        self.mergeMessageForLeaves_data = []
        self.dfs_for_merge(self.rootID, [])
        return self.mergeMessageForLeaves_data

    def convertSequenceFromVLR(self, standard: str):
        """
        :param standard: 判断前序遍历中子节点的特征
        :return: 本函数作用是前序遍历转换,转换之前先要将对象中的sequence序列装满为Node对象
        """
        if len(self.sequence) <= 1:
            print("序列缓存为空,无法转换")
        for i in range(len(self.sequence)):
            self.sequence[i] = (i, self.sequence[i])
            if i:
                self.addPoint(i, self.sequence[i][1].info)
        self.p = 2
        while len(self.sequence) > 1:
            self.addEdge(self.sequence[self.p - 1][0], self.sequence[self.p][0])
            if self.sonsOf(self.sequence[self.p - 1][0]) == 2:
                self.sequence.pop(self.p - 1)
                self.p -= 1
            if standard in self.sequence[self.p][1].info:
                self.sequence.pop(self.p)
            else:
                self.p += 1


class DecisionTree:
    def __init__(self, Table: pd.DataFrame, TrainIndex: list, TestIndex: list):
        """
        :param Table: 数据总表
        :param TrainIndex: 数据总表中的训练列的索引
        :param TestIndex: 数据总表中的测试列的索引
        """
        self.Table = Table
        self.TrainIndex = TrainIndex
        self.TestIndex = TestIndex

    def Train_train_test_split(self, test_size: float):
        """
        :param test_size: 测试集占总数据集的比值大小
        :return: 测试后的模型评测分数和训练好的模型
        """
        train = self.Table.loc[:, self.TrainIndex]
        target = self.Table.loc[:, self.TestIndex]
        xTrain_, xTest, yTrain, yTest = train_test_split(train, target, test_size=test_size)
        clf = tree.DecisionTreeClassifier(criterion='entropy')  # 参数
        clf = clf.fit(xTrain_, yTrain)
        score = clf.score(xTest, yTest)
        return score, clf

    def Train_KF(self, n_splits: int):
        """
        :param n_splits: KF交叉检验中的折数,也就是把数据分为多少类别
        :return: 测试后的模型评测分数和训练好的模型
        """
        X = self.Table.loc[:, self.TrainIndex]
        Y = self.Table.loc[:, self.TestIndex]
        KF = KFold(n_splits=n_splits)
        clf = tree.DecisionTreeClassifier(criterion='entropy')  # 参数
        for train_index, test_index in KF.split(X):
            print("TRAIN:", train_index, "TEST:", test_index, end='')
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            clf.fit(X_train, y_train)
            print("SCORE:", clf.score(X, Y))
        score = clf.score(X, Y)
        return score, clf


def Decision_Tree_Explainer(clf: tree.DecisionTreeClassifier, feature_names, class_names):
    """
    :param clf: 训练好的决策树模型
    :param feature_names: 训练决策树模型使用的数据总表中的训练列的名称
    :param class_names: 训练决策树模型使用的数据总表中的测试列的类别
    :return:
    """
    z = tree.plot_tree(clf,
                       feature_names=feature_names,
                       class_names=class_names,
                       filled=True)
    Explain = BinTree(1)
    for i in z:
        info = str(i._text).split("\n")[0]
        Explain.sequence.append(Node(info))
    Explain.convertSequenceFromVLR("entropy")
    z = Explain.mergeMessageForLeaves()
    return z


if __name__ == '__main__':
    """
    # UsageOne:
    explan = BinTree(1)
    explan.addRelationShip(1, "1", 2, "2")
    explan.addRelationShip(1, "1", 3, "3")
    explan.addRelationShip(2, "2", 4, "4")
    explan.addRelationShip(2, "2", 5, "5")
    print(explan.mergeMessageForLeaves())
    """

    """
    # UsageTwo:
    explan = BinTree(1)
    explan.sequence.append(Node("1"))
    explan.sequence.append(Node("2"))
    explan.sequence.append(Node("4#"))
    explan.sequence.append(Node("5"))
    explan.sequence.append(Node("7#"))
    explan.sequence.append(Node("9#"))
    explan.sequence.append(Node("3#"))
    explan.convertSequenceFromVLR("#")
    print(explan.mergeMessageForLeaves())
    """
    """
    # Usage Three
    from sklearn import datasets
    import pprint
    breast_cancer = datasets.load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data)
    df.columns = breast_cancer.feature_names
    df['result'] = list(breast_cancer.target)
    featureNames = list(breast_cancer.feature_names)
    Des = DecisionTree(df, featureNames, ['result']).Train_train_test_split(0.3)
    for i in range(len(featureNames)):
        featureNames[i] = str(featureNames[i])
    pprint.pprint(Decision_Tree_Explainer(Des[1], featureNames, ["0", "1"]))
    """
    print("end")

