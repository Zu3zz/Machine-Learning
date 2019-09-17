# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019/9/15 5:55 下午
# @File   : metrics.py
import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """
    计算y_true 和y_predict之间的准确度
    :param y_true: y的真值
    :param y_predict: 预测的y值
    :return: 返回正确的百分比
    """
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """
    MSE 计算平均平方差
    :param y_true: y真值
    :param y_predict: y预测值
    :return: MSE
    """
    assert len(y_true) == len(y_predict), "the length of y_true and y_predict must be the same"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """
    RMSE MSE求root
    """
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """
    MAE 使用绝对值计算 不使用差的平方和
    """
    assert len(y_true) == len(y_predict), "the length of y_true and y_predict must be the same"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """
    计算R Square
    """
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
