import numpy as np
from math import sqrt
from collections import Counter


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"
    # 函数中有随机化的部分，为了后面调试代码两次获得的随机数据一样，需要设置一个种子
    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


# 建立KNNClassifier类
class KNNClassifier:
    """初始化kNN分类器"""

    def __init__(self, k):
        # 断言k必须大于1，也就是选取的待预测附近的数据个数要大于1
        assert k >= 1, "k must be valid"

        self.k = k
        self._X_train = None
        self._y_train = None
        # self.method = 'distance'
        self.p = 2

    # 这个fit过程其实比较简单，首先建立私有的_X_train，_y_train初始值为None

    def fit(self, X_train, y_train):
        # 断言训练数据集x_train行数等于y_train的行数
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # 断言k的个数小于x_train的行数，也就是选择的‘邻居‘个数不能超过所有数据量
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        # 把用户传入的X_train, y_train赋值给私有的_X_train，_y_train
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""

        # 断言私有_X_train和_y_train都不为None，也就是说必须执行过fit拟合这一步操作才能执行本函数
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        # 断言X_predict的列，也就是特征个数等于_X_train中的特征个数，特征不相等没法预测
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        # 调用真正的预测函数_predict(x)进行处理
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""

        # 断言x行数和_X_train列数相等，也就是特征个数要相等
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        # 计算x和_X_train训练数据之间的欧式距离
        p = self.p
        distances = [np.power(np.sum(abs(x_train - x) ** p), 1.0/p)
                     for x_train in self._X_train]
        # 得到计算好的distance列表按照距离大小升序排列后对应的索引在之前列表中的位置
        nearest = np.argsort(distances)
        method = self.method
        # 将前k个索引带入_Y_train中得到对应的类别
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        # 对前k个类别投票得到最后预测结果的array
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    # 调用fit函数返回self本身的时候打印k的个数
    def __repr__(self):
        return "KNN(k=%d)" % self.k