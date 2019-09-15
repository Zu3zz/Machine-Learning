# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019/9/15 5:39 下午
# @File   : kNN.py
import numpy as np

from math import sqrt
from collections import Counter

from .metrics import accuracy_score


class KNNClassifier:
    def __init__(self, k):
        """
        初始化kNN分类器
        :param k: 近邻的个数
        """
        assert k >= 1, "k must be greater then 1"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """
        保存 X_train与y_train到self上 由于knn直接判断 所有没有fit的过程
        :param X_train: 训练数据集
        :param y_train: 训练数据集的结果
        :return:
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """
        给定数据集 进行预测
        :param X_predict:
        :return: 表示X_predict的结果向量
        """
        assert self._X_train is not None and self._y_train is not None, "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        :param x: 单个待预测数据
        :return: x的预测结果
        """
        assert x.shape[0] == self._X_train[0], "the feature number of x must equal to X_train"
        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """
        计算模型的准确率得分
        :param X_test:
        :param y_test:
        :return: 返回模型的精度
        """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k
