#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd



def distributionStatistics(data, col, train_idx, test_idx):
    # 把fips当做y值
    fips_6037 = [np.nan] * len(train_idx)
    fips_6059 = [np.nan] * len(train_idx)
    fips_6111 = [np.nan] * len(train_idx)

    train_df = data.loc[train_idx, [col, 'fips']]
    index= list(range(train_df.shape[0]))

    for i in range(5):
        col_level = {}
        for val in train_df[col].values:
            col_level[val] = [0, 0, 0]
        test_index = index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
        train_index = list(set(index).difference(test_index))
        # 记录每个col
        for i in train_index:
            temp=train_df.iloc[i]
            if temp['fips'] == 1.0:
                col_level[temp[col]][0] += 1
            if temp['fips'] == 2.0:
                col_level[temp[col]][1] += 1
            if temp['fips'] == 3.0:
                col_level[temp[col]][2] += 1

        for j in test_index:
            temp = train_df.iloc[j]
            if sum(col_level[temp[col]])!=0:
                fips_6037[j] = col_level[temp[col]][0] * 1.0 / sum(col_level[temp[col]])
                fips_6059[j] = col_level[temp[col]][1] * 1.0 / sum(col_level[temp[col]])
                fips_6111[j] = col_level[temp[col]][2] * 1.0 / sum(col_level[temp[col]])

    data.loc[train_idx, col + '_fips_6037'] = fips_6037
    data.loc[train_idx, col + '_fips_6059'] = fips_6059
    data.loc[train_idx, col + '_fips_6111'] = fips_6111


    fips_6037 = []
    fips_6059 = []
    fips_6111 = []
    col_level = {}
    for val in train_df[col].values:
        col_level[val] = [0, 0, 0]

    for i in range(train_df.shape[0]):
        temp = train_df.iloc[i]
        if temp['fips'] == 1.0:
            col_level[temp[col]][0] += 1
        if temp['fips'] == 2.0:
            col_level[temp[col]][1] += 1
        if temp['fips'] == 3.0:
            col_level[temp[col]][2] += 1

    for val in data.loc[test_idx, col].values:
        if val not in col_level.keys():
            fips_6037.append(np.nan)
            fips_6059.append(np.nan)
            fips_6111.append(np.nan)
        else:
            fips_6037.append(col_level[val][0] * 1.0 / sum(col_level[val]))
            fips_6059.append(col_level[val][1] * 1.0 / sum(col_level[val]))
            fips_6111.append(col_level[val][2] * 1.0 / sum(col_level[val]))

    data.loc[test_idx, col + '_fips_6037'] = fips_6037
    data.loc[test_idx, col + '_fips_6059'] = fips_6059
    data.loc[test_idx, col + '_fips_6111'] = fips_6111

    return data






