# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019-08-02 23:29
# @File   : kNN.py

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassfier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0]


    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None
