# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019/9/15 5:55 下午
# @File   : metrics.py
import numpy as np


def accuracy_score(y_true, y_predict):
    """
    计算y_true 和y_predict之间的准确度
    :param y_true: y的真值
    :param y_predict: 预测的y值
    :return: 返回正确的百分比
    """
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)
