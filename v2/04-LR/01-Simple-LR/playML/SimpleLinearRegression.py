# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019/9/15 6:40 下午
# @File   : SimpleLinearRegression.py
import numpy as np


class SimpleLinearRegression1:
    def __init__(self):
        """
        初始化Simple Linear Regression 模型
        """
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """
        只处理一维的向量
        :param x_train: 训练数据集
        :param y_train: 训练数据集的结果
        :return:
        """
        assert x_train.ndim == 1, "Simple Linear Regression can only solve single feature training"
        assert len(x_train) == len(y_train), "the size of x_train must be equal to y_train"

        # 下面步骤与notebook中几乎一致
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """
        :param x_predict: 待预测数据集
        :return: 表示x_predict的结果向量
        """
        assert x_predict.ndim == 1, "Simple Linear Regression can only solve single feature training"
        assert self.a_ is not None and self.b_ is not None, "Must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """
        预测单个变量
        :param x: 输入的单个变量值
        :return: 使用a_ 和 b_ 进行预测 并且返回
        """
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:
    def __init__(self):
        """
        初始化Simple Linear Regression 模型
        """
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """
        只处理一维的向量
        :param x_train: 训练数据集
        :param y_train: 训练数据集的结果
        :return:
        """
        assert x_train.ndim == 1, "Simple Linear Regression can only solve single feature training"
        assert len(x_train) == len(y_train), "the size of x_train must be equal to y_train"

        # 下面步骤与notebook中几乎一致
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """
        :param x_predict: 待预测数据集
        :return: 表示x_predict的结果向量
        """
        assert x_predict.ndim == 1, "Simple Linear Regression can only solve single feature training"
        assert self.a_ is not None and self.b_ is not None, "Must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """
        预测单个变量
        :param x: 输入的单个变量值
        :return: 使用a_ 和 b_ 进行预测 并且返回
        """
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"
