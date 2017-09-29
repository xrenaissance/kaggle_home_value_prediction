#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


###############################################################################################################
# 绝对值的预测值与实际值的误差
def mae(y, y_pred):
    return np.sum([abs(y[i] - y_pred[i]) for i in range(len(y))]) / len(y)

def Stacking1(clf, X_train, y, X_test, nfolds):
    # 交叉验证
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=0)

    # 初始化，存储训练好的模型预测的标签，由带y值的数据切分为训练集和测试集
    # 进行交叉验证后，每份测试集拼在一起，即为一份完整的数据
    train_stacker = np.zeros((X_train.shape[0],))
    # 对每轮预测的test_skf求均值即为测试数据的综合结果
    test_stacker = np.zeros((X_test.shape[0],))
    # 测试数据，不带y值，每次预测为全量
    test_skf = np.empty((nfolds, X_test.shape[0]))

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, y)):
        skf_train = X_train.loc[train_idx, :]
        skf_y = y.loc[train_idx]
        skf_test = X_train.loc[test_idx, :]

        # 训练模型
        clf.fit(skf_train, skf_y)
        train_stacker[test_idx] = clf.predict(skf_test)
        test_skf[i, :] = clf.predict(X_test)

    test_stacker[:] = test_skf.mean(axis=0)
    return train_stacker.reshape(-1, 1), test_stacker.reshape(-1, 1)


if __name__ == '__main__':
    clfs = []
    clfs.append(xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.002,
        n_estimators=100,
        silent=True,
        colsample_bytree=0.7,
        colsample_bylevel=1
    ))
