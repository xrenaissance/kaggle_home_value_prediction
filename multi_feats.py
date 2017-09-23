#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def crossFeats(data, train_idx, test_idx):
    # 使用曼哈顿距离与其它特征交叉组合
    data['mahattan_distance_bathroom'] = data['num_bathroom'] / data['mahattan_distance']
    data['mahattan_distance_bedroom'] = data['num_bedroom'] / data['mahattan_distance']
    data['mahattan_distance_area_lot'] = data['area_lot'] / data['mahattan_distance']
    data['mahattan_distance'] = data['area_total_calc'] / data['mahattan_distance']
    data['mahattan_distance_tax_property'] = data['tax_property'] / data['mahattan_distance']
    data['mahattan_distance_property_landuse'] = data['mahattan_distance'] * data['property_landuse']


    for col in ['mahattan_distance_bathroom', 'mahattan_distance_bedroom', 'mahattan_distance_area_lot',
                'mahattan_distance', 'mahattan_distance_tax_property', 'mahattan_distance_property_landuse']:
        data[col] = data[col].astype(np.float32)

    print('crossFeats: finished')
    return data
