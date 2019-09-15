# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019/9/15 6:15 下午
# @File   : model_selection.py
import numpy as np


def train_test_split(X, y, test_radio=0.2, seed=None):
    """
    :param X: 输入的所有X 
    :param y: 输入的所有y
    :param test_radio: 分割比例 
    :param seed: 随机种子
    :return: X_train,X_test,y_train,y_test
    """
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"

    if seed:
        np.random.seed(seed)
    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_radio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
