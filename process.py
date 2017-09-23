#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import math
import gc
from col_distribution import distributionStatistics
from convertAvg import convert_to_avg
from multi_feats import crossFeats
from sklearn.ensemble import RandomForestRegressor



feats_not_use = ['transaction_date', 'building_class_type_id', 'deck_type', 'story_type', 'area_liveperi_finished',
                 'area_base', 'area_basement', 'pool_type_7', 'pool_type_10', 'pool_type_2',
                 'raw_census_tract_and_block', 'tax_delinquency_year', 'flag_tax_delinquency', 'assessment_year',
                 'flag_fireplace', 'flag_tub', 'area_pool', 'property_zoing', 'property_county', 'num_bathroom_grp_county']

feats_not_use.extend(['num_fireplace', 'num_bathroom_grp_county', 'area_firstfloor', 'median_num_bedroom'
                      'median_num_bathroom', 'num_unit', 'area_total_finished', 'area_live_finished'])

feats_not_use.extend(['fips', 'num_room', 'tax_structure_num_room', 'tax_property_num_room', 'bedroom_per_room',
                      'num_75_per_bath', 'num_75_per_bed', 'num_story', 'region_county', 'area_shed',
                      'building_type', 'building_style', 'num_bath'])


################################################################################################
'''
处理变量: air_conditioning_type(airconditioningtypeid)，空调类别，数值型
air_conditioning_type: 空调类别，当fips为6111时，全为缺失值
考虑把fips为6111的缺失归为一类
'''
def preprocessAirType(data):
    # train_data.loc[train_data['fips']==6011, 'air_conditioning_type'].isnull().value_counts()
    data.loc[data['fips']==6011, 'air_conditioning_type'].fillna(-1, inplace=True)
    data['air_conditioning_type'].fillna(0, inplace=True)

    data['air_conditioning_type'] = data['air_conditioning_type'].astype(np.uint32)

    print('preprocessAirType: finished')
    return data

################################################################################################
'''
处理变量: num_bathroom、num_bathroom_calc、num_bath、num_75_bath
num_bathroom: 浴室数量，包括小浴室
num_bathroom_calc: 浴室数量，与上面num_bathroom相同，但有缺失值
num_bath: 浴室数量，有完整的浴室(水槽，淋浴+浴缸和卫生间)，没有包含小数点，向下取整
num_75_bath: 淋浴+水槽+卫生间
'''
# 处理浴室数量
def preprocessNumBathroom(data, train_idx, test_idx):
    # 缺失值填充众数
    data['num_bathroom'].fillna(2.0, inplace=True)
    data['num_bedroom'].fillna(3.0, inplace=True)

    bathroom_cnt = data['num_bathroom'].value_counts()
    data['bathroom_cnt'] = data['num_bathroom'].map(lambda x: bathroom_cnt[x])

    # 离群点处理
    #data.loc[data['num_bathroom'] > 10, 'num_bathroom'] = 10
    # 占比越大，表明浴室较少，占比越小，表明浴室多，如每间卧室都配有浴室，则为1/2
    data['bathroom_per_room'] = data['num_bathroom'] / (data['num_bedroom'] + data['num_bathroom'])
    # 缺失值填充0
    data['num_75_bath'].fillna(0, inplace=True)

    for col in ['num_bathroom', 'num_bedroom', 'num_75_bath']:
        data[col] = data[col].astype(np.uint8)

    print('preprocessNumBathroom: finished')
    return data

################################################################################################
# 处理卧室数量
def preprocessNumBedroom(data, train_idx, test_idx):
    # 填充0或众数3
    data['num_bedroom'].fillna(3.0, inplace=True)
    bedroom_cnt = data['num_bedroom'].value_counts()
    data['bedroom_cnt'] = data['num_bedroom'].map(lambda x: bedroom_cnt[x])

    # 离群点处理
    #data.loc[data['num_bedroom'] > 10, 'num_bedroom'] = 10

    # 浴室数 * 卧室数
    data['num_bath_bed'] = data['num_bathroom'] * data['num_bedroom']
    # 卧室 - 浴室
    data['bed_bath_diff'] = data['num_bedroom'] - data['num_bathroom']
    # 浴室和卧室的比值
    data['bed_bath_per'] = data['num_bedroom'] / (data['num_bathroom'] + 1.0)
    data['bathroom_per_room_0'] = data['num_bathroom'] / (data['num_bedroom'] + data['num_bathroom'] + 1.0)

    for col in ['num_bath_bed', 'bed_bath_diff', 'bed_bath_per', 'bathroom_per_room_0']:
        data[col] = data[col].astype(np.float32)

    print('preprocessNumBedroom: finished')
    return data

################################################################################################
'''
处理fips，6037 Los Angeles，6059 Orange County，6111 Ventura County
使用有序变量，与y值比较，对模型无效果，丢弃
'''
def preprocessFips(data):
    # fips缺失的同时，经度和纬度也缺失
    # 填充众数，重编码
    data['fips'].fillna(6037.0, inplace=True)
    data.loc[data['fips'] == 6037, 'fips'] = 1.0
    data.loc[data['fips'] == 6059, 'fips'] = 2.0
    data.loc[data['fips'] == 6111, 'fips'] = 3.0

    data['fips'] = data['fips'].astype(np.uint8)
    print('preprocessFips: finished')
    return data

################################################################################################
'''
处理变量: region_county、region_city、region_zip、region_neighbor
region_county: 3101对应fips的6037，1286对应fips的6059，2061对应fips的6111
region_city: 物业所在的城市，缺失可用邮编求出
region_zip: 物业的邮政编码，共有389个，排序方法
region_neighbor: 物业所在的街区
'''
def preprocessRegion(data, train_idx, test_idx):
    # 缺失值，使用fips==6037的region_zip的均值
    train = data.loc[train_idx, :].copy()
    test = data.loc[test_idx, :].copy()
    train_mean = train.loc[train['fips']==1, 'region_zip'].mode()
    test_mean = test.loc[test['fips']==1, 'region_zip'].mode()
    train.loc[train['region_zip'].isnull(), 'region_zip'] = float(train_mean)
    test.loc[test['region_zip'].isnull(), 'region_zip'] = float(test_mean)
    data.loc[train_idx, :] = train
    data.loc[test_idx, :] = test

    # 第一种方案，汇总邮编
    # 第二种方法，按照某种规则重新排序，如经纬度的欧几里的距离
    # 缺失值填充3101，对应fips的填充6037
    data['region_county'].fillna(3101.0, inplace=True)
    # 使用邮编region_zip来填充region_city的缺失值，如region_zip的邮编为96974，其region_city为37086
    grp_city_zip = data[['region_city', 'region_zip']]
    grp_df = grp_city_zip.groupby(['region_zip']).apply(pd.DataFrame.mode).reset_index(drop=True)
    # 转为字典型，进行匹配
    grp_zip_dict = {}
    for i in range(len(grp_df)):
        grp_zip_dict[grp_df.loc[i, 'region_zip']] = grp_df.loc[i, 'region_city']

    data.loc[np.logical_and(data['region_city'].isnull(), data['region_zip'].notnull()), 'region_city'] = data.loc[
            np.logical_and(data['region_city'].isnull(), data['region_zip'].notnull()), 'region_zip'].map(lambda x: grp_zip_dict[x])

    col_cnt = dict(data['region_zip'].value_counts())
    data['region_zip_cnt'] = data['region_zip'].apply(lambda x: col_cnt[x])

    # 对region_neighbor进行缺失值填充
    grp_neighbor_zip = data[['region_neighbor', 'region_zip']]
    grp_df2 = grp_neighbor_zip.groupby(['region_zip']).apply(pd.DataFrame.mode).reset_index(drop=True)
    # 转为字典型，进行匹配
    grp_zip_dict2 = {}
    for i in range(len(grp_df2)):
        grp_zip_dict2[grp_df2.loc[i, 'region_zip']] = grp_df2.loc[i, 'region_neighbor']

    data.loc[np.logical_and(data['region_neighbor'].isnull(), data['region_zip'].notnull()), 'region_neighbor'] = data.loc[
            np.logical_and(data['region_neighbor'].isnull(), data['region_zip'].notnull()), 'region_zip'].map(lambda x: grp_zip_dict2[x])

    grp_neighbor_city = data[['region_neighbor', 'region_city']]
    grp_df3 = grp_neighbor_city.groupby(['region_city']).apply(pd.DataFrame.mode).reset_index(drop=True)
    # 转为字典型，进行匹配
    grp_zip_dict3 = {}
    for i in range(len(grp_df3)):
        grp_zip_dict3[grp_df3.loc[i, 'region_city']] = grp_df3.loc[i, 'region_neighbor']

    data.loc[np.logical_and(data['region_neighbor'].isnull(), data['region_city'].notnull()), 'region_neighbor'] = data.loc[
            np.logical_and(data['region_neighbor'].isnull(), data['region_city'].notnull()), 'region_city'].map(lambda x: grp_zip_dict3[x])

    data['region_neighbor'].fillna(0.0, inplace=True)
    gc.collect()

    print('preprocessRegion: finished')
    return data

################################################################################################
# 处理纬度和经度变量
def preprocessLocation(data, train_idx, test_idx):
    train = data.loc[train_idx, :].copy()
    test = data.loc[test_idx, :].copy()
    # 经纬度缺失的同时，fips也缺失，对fips填众数6037，经纬度则取fips对应的经纬度的均值
    train_lat_mean = train.loc[train['fips']==1.0, 'latitude'].mean()
    test_lat_mean = test.loc[test['fips']==1.0, 'latitude'].mean()
    train.loc[train['latitude'].isnull(), 'latitude'] = float(train_lat_mean)
    test.loc[test['latitude'].isnull(), 'latitude'] = float(test_lat_mean)

    train_lon_mean = train.loc[train['fips']==1.0, 'longitude'].mean()
    test_lon_mean = test.loc[test['fips']==1.0, 'longitude'].mean()
    train.loc[train['longitude'].isnull(), 'longitude'] = float(train_lon_mean)
    test.loc[test['longitude'].isnull(), 'longitude'] = float(test_lon_mean)

    data.loc[train_idx, :] = train
    data.loc[test_idx, :] = test

    # 经度和纬度相加
    data['latitude_plus_longitude'] = data['latitude'] + data['longitude']
    data['latitude_multi_longitude'] = data['latitude'] * data['longitude']
    # 维度和经度的欧几里得距离
    data['long_lat_distance'] = (data['longitude'] - data['longitude'].mean())**2 + (data['latitude'] - data['latitude'].mean())**2
    data['long_lat_distance'] = data['long_lat_distance'].map(lambda x: math.sqrt(x))

    # 每个经纬度分类一下(1平方千米一个类)
    #data['jwd_class'] = map(lambda x, y: (int(x*100)%100)*100 + (int(-y*100)%100), data['latitude'], data['longitude'])

    # 聚类
    train_x = data.loc[train_idx, :][['latitude', 'longitude']]
    test_x = data.loc[test_idx, :][['latitude', 'longitude']]
    # 聚类，将每个房子划分到一块区域中，然后算出每个类别的中点坐标，计算曼哈顿距离(或欧几里得距离)
    kmeans_cluster = KMeans(n_clusters=20)
    # 拟合训练集，并预测
    clf = kmeans_cluster.fit(train_x)
    val_pred = clf.predict(pd.concat([train_x, test_x]))
    # 聚类后，给每个parcel_id一个类别，0~19类，即相似位置的清单数为一类
    parcel_id_dict = dict(zip(data['parcel_id'], val_pred))
    data['cenroid'] = data['parcel_id'].apply(lambda x: parcel_id_dict[x])
    # 曼哈顿距离，计算所有数据的经度和维度的均值，即经度和维度的中心点
    center = [data.loc[train_idx, :]['latitude'].mean(), data.loc[train_idx, :]['longitude'].mean()]
    # center[0]为latitude的均值，center[1]为longitude的均值
    data['mahattan_distance'] = abs(data['latitude'] - center[0] + abs(data['longitude'] - center[1]))
    data['euclidean_distance'] = ((data['latitude'] - center[0])**2 + (data['longitude'] - center[1])**2)**0.5
    # 经纬度 / 卧室、浴室
    data['latitude_per_bath'] = data['latitude'] / data['num_bathroom'][data['num_bathroom'] != 0]
    data['longitude_per_bath'] = data['longitude'] / data['num_bathroom'][data['num_bathroom'] != 0]
    data['latitude_per_bed'] = data['latitude'] / data['num_bedroom'][data['num_bedroom'] != 0]
    data['longitude_per_bed'] = data['longitude'] / data['num_bedroom'][data['num_bedroom'] != 0]

    # 缺失值填充0
    for col in ['latitude_per_bath', 'longitude_per_bath', 'latitude_per_bed', 'longitude_per_bed']:
        data.loc[:, col].fillna(0.0, inplace=True)
        data[col] = data[col].astype(np.float32)

    print('preprocessLocation: finished')
    return data

################################################################################################
'''
处理变量: raw_census_tract_and_block、census_tract_and_block
raw_census_tract_and_block: 由FIPS Code (6037) - Tract Number (8002.04) - And block Number (1)组成
census_tract_and_block: 与raw_census_tract_and_block一样，但有缺失值
'''
def preprocessCensusBlock(data):
    # 拆分成两部部分，fips不用拆分，前面已经有了
    data['tract_number'] = data['raw_census_tract_and_block'].map(lambda x: str(x)[4:11])
    data['tract_number'] = data['tract_number'].map(lambda x: 0 if x == '' else float(x))

    data['tract_block'] = data['raw_census_tract_and_block'].map(lambda x: str(x)[4:])
    data['tract_block'] = data['tract_block'].map(lambda x: 0 if x == '' else float(x))

    data['block_number'] = data['raw_census_tract_and_block'].map(lambda x: str(x)[11:12])
    data['block_number'] = data['block_number'].map(lambda x: 0 if x == '' else float(x))

    Le = LabelEncoder()
    data['raw_census_tract_and_block_encoder'] = data['raw_census_tract_and_block'].astype(str)
    data['raw_census_tract_and_block_encoder'] = Le.fit_transform(data['raw_census_tract_and_block_encoder'])

    for col in ['tract_number', 'tract_block', 'block_number', 'raw_census_tract_and_block_encoder']:
        data[col] = data[col].astype(np.float32)

    print('preprocessCensusBlock: finished')
    return data

################################################################################################
'''
处理变量: area_total_calc、area_base、area_live_finished、area_liveperi_finished、area_total_finished、area_unknown、area_firstfloor_finished
area_total_calc: 总居住面积，其值由area_live_finished和area_total_finished组成
area_live_finished: 完全生活区面积，与area_total_calc值一样
area_total_finished: 总面积，与area_total_calc值一样
area_firstfloor_finished: 第一楼层的居住面积
area_firstfloor: 第一楼层的居住面积，与area_firstfloor_finished大部分相同，只有小部分值不同
area_liveperi_finished: 周边生活区，缺失比率99.9%，丢弃
area_base: 地下室完成和未完成面积，缺失比率99.5%，丢弃
area_basement: 生活区低于地面的面积，缺失比率99.9%，丢弃
area_lot: 地块面积平方
'''
def preprocessFinishedSquare(data):
    # 离群点处理
    ulimit = 5100
    data.loc[data['area_total_calc'] > ulimit, 'area_total_calc'] = ulimit
    dlimit = 100
    data.loc[data['area_total_calc'] < dlimit, 'area_total_calc'] = dlimit

    # 缺失值处理，因为tax_structure加上tax_land等于tax_value_num，用于线性回归预测
    data.loc[data['tax_structure'].isnull(), 'tax_structure'] = data.loc[data['tax_structure'].isnull(), 'tax_value_num'] - data.loc[
                                                                data['tax_structure'].isnull(), 'tax_land']
    data['tax_structure'].fillna(0.0, inplace=True)

    # 使用线性回归对缺失值进行预测
    if np.any(data['area_total_calc'].isnull()):
        Lr = LinearRegression()
        train_X = data.loc[data['area_total_calc'].notnull(), ['num_bathroom', 'num_bedroom', 'tax_structure']]
        train_y = data.loc[data['area_total_calc'].notnull(), 'area_total_calc']
        test_X = data.loc[data['area_total_calc'].isnull(), ['num_bathroom', 'num_bedroom', 'tax_structure']]
        # 数据标准化处理
        stda = StandardScaler()
        train_X = stda.fit_transform(train_X)
        test_X = stda.transform(test_X)
        # 拟合建模
        Lr.fit(train_X, train_y)
        # 分数0.64
        print('area_total_calc: ', Lr.score(train_X, train_y))
        data.loc[data['area_total_calc'].isnull(), 'area_total_calc'] = Lr.predict(test_X)

    # 每间房的大小
    data['area_per_bed_bath'] = data['area_total_calc'] / (data['num_bathroom'] + data['num_bedroom'] + 1.0)
    # 每间卧室大小
    data['area_per_bedroom'] = data['area_total_calc'] / (data['num_bedroom'] + 1.0)
    # 每间浴室大小
    data['area_per_bathroom'] = data['area_total_calc'] / (data['num_bathroom'] + 1.0)
    # 每个经度和纬度的大小面积
    data['area_per_latitude'] = data['area_total_calc'] / (data['latitude'] + 1.0)
    data['area_per_longitude'] = data['area_total_calc'] / (data['longitude'] + 1.0)
    # 每个room的大小
    data['area_per_room'] = data['area_total_calc'] / (data['num_room'] + 1.0)

    # 缺失值填充0
    data['area_lot'].fillna(0.0, inplace=True)
    # 离群点处理
    ulimit = 456960
    data.loc[data['area_lot'] > ulimit, 'area_lot'] = ulimit
    # 每间房的大小
    data['area_lot_room'] = data['area_lot'] / (data['num_bathroom'] + data['num_bedroom'] + 1.0)
    # 每间卧室大小
    data['area_lot_bedroom'] = data['area_lot'] / (data['num_bedroom'] + 1.0)
    # 每间浴室大小
    data['area_lot_bathroom'] = data['area_lot'] / (data['num_bathroom'] + 1.0)
    # 每个经度和纬度的大小
    data['area_lot_latitude'] = data['area_lot'] / (data['latitude'] + 1.0)
    data['area_lot_longitude'] = data['area_lot'] / (data['longitude'] + 1.0)
    #
    data['area_lot_total_calc'] = data['area_lot'] / data['area_total_calc']

    # error in calculation of the finished living area of home
    data['area_live_finished'].fillna(0.0, inplace=True)
    data['N-LivingAreaError'] = data['area_live_finished']/data['area_total_calc']

    # proportion of living area
    data['area_total_finished'].fillna(0.0, inplace=True)
    data['N-LivingAreaProp2'] = data['area_live_finished'] / (data['area_total_finished'] + 1.0)

    # Amout of extra space
    data['extra_space'] = data['area_lot'] - data['area_total_calc']
    data['extra_space2'] = data['area_total_finished'] - data['area_live_finished']

    median_lst = ['num_bedroom', 'num_bathroom', 'area_lot']
    for col in median_lst:
        # 对col列进行聚合，并求其area的均值，如num_bedroom聚合，求出num_bedroom为x间房时的面积
        median_area = data[[col, 'area_total_calc']].groupby(col)['area_total_calc'].median()
        # 字典映射到data[col]，键为data[col]，值为median_area的值
        median_area = median_area[data[col]].values.astype(float)
        #data['median_' + col] = median_area
        # 面积除以该特征聚合后的中位数面积，如卧室数的面积/ 聚合后的卧室数的中位数面积
        data['ratio_' + col] = data['area_total_calc'] / median_area
        # 对中位数取对数，底数为e
        #data['median_' + col] = data['median_' + col].apply(lambda x: np.log(x))

    median_lst = ['area_lot', 'area_total_calc']
    for col in median_lst:
        # 对col列进行聚合，并求其area的均值，如num_bedroom聚合，求出num_bedroom为x间房时的面积
        mean_area = data[[col, 'region_zip']].groupby(col)['region_zip'].mean()
        # 字典映射到data[col]，键为data[col]，值为median_area的值
        mean_area = mean_area[data[col]].values.astype(float)
        data['mean_' + col] = mean_area
        # 面积除以该特征聚合后的中位数面积，如卧室数的面积/ 聚合后的卧室数的中位数面积
        data['abs_ratio_' + col] = abs(data[col] -  data['mean_' + col]) / data['mean_' + col]

    for col in ['area_total_calc', 'tax_structure', 'area_per_bed_bath', 'area_per_bedroom', 'area_per_bathroom',
                'area_per_latitude', 'area_per_longitude', 'area_per_room', 'area_lot', 'area_lot_room',
                'area_lot_bedroom', 'area_lot_bathroom', 'area_lot_latitude', 'area_lot_longitude', 'area_lot_total_calc',
                'area_live_finished', 'N-LivingAreaError', 'area_total_finished', 'N-LivingAreaProp2', 'extra_space',
                'extra_space2', 'ratio_num_bedroom', 'ratio_num_bathroom', 'ratio_area_lot',
                'mean_area_lot', 'mean_area_total_calc', 'abs_ratio_area_lot', 'abs_ratio_area_total_calc',
                'area_basement', 'area_base', 'area_liveperi_finished']:
        data[col] = data[col].astype(np.float32)

    print('preprocessFinishedSquare: finished')
    return data

################################################################################################
# 交叉组合特征，统计property_county列下的area_total_calc、num_bathroom、num_bedroom的均值
def group_with_property_county(data, feat):
    mean_feat_dict = dict(data.groupby('property_county')[feat].mean())
    data[feat + '_grp_county'] = data['property_county'].apply(lambda x: mean_feat_dict[x])
    data[feat + '_grp_county'] = data[feat + '_grp_county'].astype(np.float32)

    return data

'''
处理变量: property_county、property_landuse、property_zoing
property_county: 县级土地使用代码，即县级分区
property_landuse: 土地用途类型
property_zoing: 用途分区说明，编码型
'''
def preprocessProperty(data):
    # 缺失值填众数，因0100占训练集的1/3
    data['property_county'].fillna('0100', inplace=True)

    # 方案一，统计该列的值，计数
    col_cnt = dict(data['property_county'].value_counts())
    data['property_county_cnt'] = data['property_county'].apply(lambda x: col_cnt[x])
    # 方案二， 使用类似NLP里的最大词频方法，选择前100个次数最多的值
    # 方案三，使用贝叶斯统计

    # 编码转换
    Le = LabelEncoder()
    data['property_county_encoder'] = data['property_county'].astype(str)
    data['property_county_encoder'] = Le.fit_transform(data['property_county_encoder'])

    # 土地用途类型，缺失值填众数
    data['property_landuse'].fillna(261.0, inplace=True)

    # 用途区分说明，缺失值太多，将其归为一类
    data['property_zoing'].fillna('other', inplace=True)
    # 统计该值的列，计数
    col_cnt = dict(data['property_zoing'].value_counts())
    data['property_zoing_cnt'] = data['property_zoing'].apply(lambda x: col_cnt[x])

    data['property_zoing_encoder'] = data['property_zoing'].astype(str)
    data['property_zoing_encoder'] = Le.fit_transform(data['property_zoing_encoder'])

    # 交叉组合特征，统计property_county列下的area_total_calc、num_bathroom、num_bedroom的均值
    for col in ['area_total_calc', 'num_bedroom', 'num_unit', 'latitude', 'longitude', 'tax_structure', 'building_quality_type']:
        data = group_with_property_county(data, col)

    for col in['property_landuse', 'property_zoing_cnt', 'property_zoing_encoder',
               'property_county_encoder', 'property_county_cnt']:
        data[col] = data[col].astype(np.uint8)

    del data['property_zoing'], data['property_county']
    gc.collect()

    print('preprocessProperty: finished')
    return data

################################################################################################
'''
处理变量: num_room
num_room: 主要住所的房间总数，个人理解为卧室房间数+浴室房间数+其它房间数，
但从数据来看，num_room有2/3的值为0，放到模型不佳，与其它特征交叉组合也没什么用，暂时丢弃
'''
def preprocessNumRoom(data):
    # 缺失值填充0
    data['num_room'].fillna(0, inplace=True)

    # num_room是否为0
    #data['flag_room_zero'] = data['num_room'].map(lambda x: True if x == 0 else False)
    #data['room_bath_bed_ratio'] = data['num_room'] / (data['num_bedroom'] + data['num_bathroom'] + 1.0)

    data['num_room'] = data['num_room'].astype(np.uint8)
    #data['room_bath_bed_ratio'] = data['room_bath_bed_ratio'].astype(np.float32)

    print('preprocessNumRoom: finished')
    return data

################################################################################################
'''
处理变量: num_unit
num_unit: 内置结构的单位数
放入模型效果不佳，且值为1占据1/2，另有大半为缺失值，暂时丢弃
'''
def preprocessNumUnit(data):
    # 缺失值填充0，当做另一类
    data['num_unit'].fillna(0, inplace=True)
    '''
    # 离群点处理
    ulimit = 30
    data.loc[data['num_unit'] > ulimit, 'num_unit'] = ulimit
    '''
    data['num_unit'] = data['num_unit'].astype(np.uint8)

    print('preprocessNumUnit: finished')
    return data

################################################################################################
'''
处理阳台类别，缺失比率99.3%，且为同一值，丢弃
'''
def preprocessDeckType(data):
    data['deck_type'].fillna(0, inplace=True)
    data['deck_type'] = data['deck_type'].astype(np.uint8)

    print('preprocessDeckType: finished')
    return data

################################################################################################
'''
处理变量: num_story、story_type
num_story: 多少层，丢弃
story_type: 多层房屋的楼层类型，训练集和测试集只有一种类型，丢弃
'''
def preprocessStory(data):
    # 缺失值填充0
    data['num_story'].fillna(0, inplace=True)
    data['num_story'] = data['num_story'].astype(np.uint8)

    data['story_type'].fillna(0, inplace=True)
    data['story_type'] = data['story_type'].astype(np.uint8)

    print('preprocessStory: finished')
    return data

################################################################################################
'''
处理变量: num_fireplace、flag_fireplace
num_fireplace: 壁炉数量，有
flag_fireplace: 是否有壁炉，
一个问题没壁炉但有壁炉数量，没壁炉数量但有壁炉
'''

def preprocessFirePlace(data):
    data['num_fireplace'].fillna(0, inplace=True)
    data['num_fireplace'] = data['num_fireplace'].astype(np.uint8)

    del data['flag_fireplace']
    gc.collect()

    print('preprocessFirePlace: finished')
    return data

################################################################################################
'''
处理变量: num_garage、area_garage
num_garage: 车库数量
area_garage: 车库面积平方
'''
def preprocessGarage(data):
    # 车库数量和车库面积同时缺失，填充0
    data['num_garage'].fillna(0.0, inplace=True)
    data['area_garage'].fillna(0.0, inplace=True)
    # 离群点处理
    ulimit = 920
    data.loc[data['area_garage'] > ulimit, 'area_garage'] = ulimit
    dlimit = 0
    data.loc[data['area_garage'] < dlimit, 'area_garage'] = dlimit

    # 车库数量*平方
    #data['total_area_garage'] = data['num_garage'] * data['area_garage']
    #data['area_per_garage'] = data['num_garage'] / (data['area_garage'] + 1.0)
    # 车库面积除以总面积
    #data['area_garage_per_total'] = data['area_garage'] / data['area_total_calc']

    data['num_garage'] = data['num_garage'].astype(np.uint8)
    for col in ['area_garage']:
        data[col] = data[col].astype(np.float32)

    print('preprocessGarage: finished')
    return data

################################################################################################
'''
处理变量: flag_tub、pool_type_10、pool_type_2、pool_type_7、num_pool、area_pool
flag_tub: 是否有hot tub或spa，为pool_type_10+pool_type_2的总和
pool_type_10: 没泳池但有hot tub或spa
pool_type_2: 有泳池，且有hot tub或spa
pool_type_7: 有泳池没有hot tub
整合成一列，有序变量，pool_type_7为1，pool_type_10为2，pool_type_2为3，都没则为0
num_pool: 泳池数量，其和为pool_type_7和pool_type_2的加总，其值为1，只有一个泳池，无法做加权
area_pool: 泳池面积，有值的为969个，在pool_type_7中，但仍有15728个是没有值的
'''
def preprocessPool(data):
    data['pool_type'] = 0.0
    data.loc[data['pool_type_7'].notnull(), 'pool_type'] = 1.0
    data.loc[data['pool_type_10'].notnull(), 'pool_type'] = 2.0
    data.loc[data['pool_type_2'].notnull(), 'pool_type'] = 3.0

    # 离群点处理
    ulimit = 950
    data.loc[data['area_pool'] > ulimit, 'area_pool'] = ulimit
    # 缺失值填充0
    data['area_pool'].fillna(0.0, inplace=True)

    # 泳池面积乘以泳池类别，进行加权，没用丢弃
    #data['weight_pool'] = data['pool_type'] * data['area_pool']

    for col in ['pool_type', 'area_pool']:
        data[col] = data[col].astype(np.uint8)

    del data['pool_type_7'], data['pool_type_10'], data['pool_type_2']
    gc.collect()

    print('preprocessPool: finished')
    return data

################################################################################################
'''
处理变量: area_patio、area_shed
area_patio: 天井在院子里
area_shed: 仓库棚/建筑在院子里，丢弃
'''
def preprocessYard(data):
    # 缺失值填充0
    data['area_patio'].fillna(0, inplace=True)
    data['area_shed'].fillna(0, inplace=True)
    # 仓库面积与庭院面积的比率
    #data['shed_patio_ratio'] = data['area_shed'] / (data['area_patio'] + 1.0)
    # 占总面积的比率
    for col in ['area_patio', 'area_shed']:
        data[col] = data[col].astype(np.uint8)

    print('preprocessYard: finished')
    return  data

################################################################################################
# 6037 Los Angeles，6059 Orange County，6111 Ventura County
'''
处理加热类别，观察到Ventura County(6111)的加热系统都为空，查看百科其冬天气温不需要加热系统

'''
def preprocessHeatingType(data):
    # 对于缺失值，先填充None(13)，可考虑通过经纬度判断是否需要加热系统
    data['heating_type'].fillna(0, inplace=True)
    # 是否有加热系统
    #data['flag_heating'] = data['heating_type'].map(lambda x: False if math.isnan(x) else True)
    # 汇总归为四类，Central(2)、Floor/Wall(7)、Other和None(13)
    # 重新归类后，Other为3，None为1
    data['heating_type'] = data['heating_type'].astype(np.uint8)

    print('preprocessHeatingType: finished')
    return data

################################################################################################
'''
处理变量: tax_value_num、tax_structure、tax_land、tax_property、assessment_year、flag_tax_delinquency、tax_delinquency_year
tax_value_num: 总税额评估值，其值为tax_structure和tax_land相加的和
tax_structure: 建筑结构评估值
tax_land: 土地面积的评估价值
tax_property: 该评估年度的财产税总额
assessment_year: 财产税评估年度，训练集为单一值2015年，丢弃
flag_tax_delinquency: 截至2015年，财产税是否到期，缺失比率为98.0%，丢弃或考虑与其它列组合
tax_delinquency_year: 未缴财产税到期的年份，缺失比率达98.0%，丢弃或考虑与其它列组合
'''
def preprocessTax(data):
    # tax_structure的缺失值已在preprocessFinishedSquare进行处理，用于线性回归
    data['tax_land'].fillna(0, inplace=True)
    data['tax_value_num'].fillna(0, inplace=True)
    # 对于tax_property的缺失值使用线性回归进行预测
    Lr = LinearRegression()
    train_X = data.loc[data['tax_property'].notnull(), ['tax_land', 'tax_value_num']]
    train_y = data.loc[data['tax_property'].notnull(), 'tax_property']
    test_X = data.loc[data['tax_property'].isnull(), ['tax_land', 'tax_value_num']]
    # 数据标准化处理
    stda = StandardScaler()
    train_X = stda.fit_transform(train_X)
    test_X = stda.transform(test_X)
    # 拟合建模
    Lr.fit(train_X, train_y)
    # 分数0.90
    print('tax_property: ', Lr.score(train_X, train_y))
    data.loc[data['tax_property'].isnull(), 'tax_property'] = Lr.predict(test_X)

    # 对tax_land和tax_structure进行离群点处理，然后相加得到tax_value_num
    # 建筑评估占总的评估比率
    data['tax_structure_value_num_ratio'] = data['tax_structure'] / (data['tax_value_num'] + 1.0)
    # 土地评估占总的评估比率
    data['tax_land_value_num_ratio'] = data['tax_land'] / (data['tax_value_num'] + 1.0)
    # 财产税占总额的比率
    data['tax_property_value_num_ratio'] = data['tax_property'] / (data['tax_value_num'] + 1.0)

    # 交叉组合特征
    col_lst = ['area_total_calc', 'area_lot', 'num_bathroom', 'num_bedroom', 'num_75_bath', 'latitude',
               'longitude', 'mahattan_distance', 'air_conditioning_type', 'built_year', 'pool_type',
               'region_zip', 'tract_number', 'block_number', 'raw_census_tract_and_block_encoder', 'heating_type']
    for col in col_lst:
        data[col + '_property'] = data[col] / (data['tax_property'])
        data[col + '_property'] = data[col + '_property'].astype(np.float32)
        data[col + '_structure'] = data[col] / (data['tax_structure'] + 0.001)
        data[col + '_structure'] = data[col + '_structure'].astype(np.float32)
        data[col + '_value'] = data[col] / (data['tax_value_num'] + 0.001)
        data[col + '_value'] = data[col + '_value'].astype(np.float32)

    data['room_property'] = (data['num_bathroom'] + data['num_bedroom']) / data['tax_property']
    data['room_structure'] = (data['num_bathroom'] + data['num_bedroom']) / (data['tax_structure'] + 0.001)
    data['room_value_num'] = (data['num_bathroom'] + data['num_bedroom']) / (data['tax_value_num'] + 0.001)

    # 财产税占建筑评估的比率
    data['tax_structure_property_ratio'] = data['tax_structure'] / data['tax_property']
    # 财产税占土地评估的比率
    data['tax_land_property_ratio'] = data['tax_land'] / data['tax_property']
    # 土地面积的评估值 * 土地面积

    #
    data['property_value_ratio'] = data['tax_value_num'] / data['tax_property']

    for col in ['tax_land', 'tax_value_num']:
        data[col] = data[col].astype(np.uint8)

    for col in ['property_value_ratio', 'tax_structure_property_ratio',
                'tax_land_property_ratio', 'room_property', 'room_structure', 'room_value_num',
                'tax_property_value_num_ratio', 'tax_land_value_num_ratio',
                'tax_structure_value_num_ratio', 'tax_property']:
        data[col] = data[col].astype(np.float32)

    del data['tax_delinquency_year'], data['flag_tax_delinquency'], data['assessment_year']
    gc.collect()

    print('preprocessTax: finished')
    return data

################################################################################################
# 处理变量建造年份
def preprocessBuiltYear(data):
    data['built_year'].fillna(0.0, inplace=True)

    for col in ['built_year']:
        data[col] = data[col].astype(np.uint8)

    data['area_per_year'] = data['built_year'] / data['area_total_calc']

    data['area_per_year'] = data['area_per_year'].astype(np.float32)

    print('preprocessBuiltYear: finished')
    return data

################################################################################################
'''
处理变量: building_class_type_id、building_quality_type
building_class_type_id: 建筑类别，只有16个有值为4，其余为缺失值，丢弃
building_quality_type: 建筑质量，值越小质量越好，相反值越大质量越差
building_type: 99%都为缺失值，丢弃
building_style: 99%都为缺失值，丢弃
'''
def preprocessBuilding(data):
    # 缺失值填充0
    data['building_type'].fillna(0, inplace=True)
    data['building_style'].fillna(0, inplace=True)
    '''
    # 缺失值使用随机森林进行预测
    clf = RandomForestRegressor(n_estimators=500, max_depth=6, oob_score=True)
    train_X = data.loc[data['building_quality_type'].notnull(), :].fillna(0)
    train_y = data.loc[data['building_quality_type'].notnull(), 'building_quality_type']
    test_X = data.loc[data['building_quality_type'].isnull(), :].fillna(0)
    clf.fit(train_X, train_y)
    print('building_quality_type: ', clf.oob_score_)
    data.loc[data['building_quality_type'].isnull(), 'building_quality_type'] = clf.predict(test_X)
    '''
    Lr = LinearRegression()
    train_X = data.loc[data['building_quality_type'].notnull(), ['area_total_calc', 'built_year', 'tax_land',
                                'tax_property', 'air_conditioning_type', 'num_bathroom', 'num_bedroom',
                                'building_type', 'building_style', 'num_story', 'deck_type', 'story_type',
                                'num_fireplace', 'pool_type', 'area_pool']]
    train_y = data.loc[data['building_quality_type'].notnull(), 'building_quality_type']
    test_X = data.loc[data['building_quality_type'].isnull(), ['area_total_calc', 'built_year', 'tax_land',
                                'tax_property', 'air_conditioning_type', 'num_bathroom', 'num_bedroom',
                                'building_type', 'building_style', 'num_story', 'deck_type', 'story_type',
                                'num_fireplace', 'pool_type', 'area_pool']]
    # 数据标准化
    stda = StandardScaler()
    train_X = stda.fit_transform(train_X)
    test_X = stda.transform(test_X)
    # 拟合
    Lr.fit(train_X, train_y)
    # 分数0.23
    print('building_quality_type: ', Lr.score(train_X, train_y))
    data.loc[data['building_quality_type'].isnull(), 'building_quality_type'] = Lr.predict(test_X)

    del data['building_type'], data['building_style'], data['building_class_type_id'], data['deck_type'], data['num_story'], data['story_type']
    del train_X, train_y, test_X
    gc.collect()

    print('preprocessBuilding: finished')
    return data

################################################################################################
# 特征工程
def coreProcess(data, train_idx, test_idx):
    data = preprocessAirType(data)
    data = preprocessNumBathroom(data, train_idx, test_idx)
    data = preprocessNumBedroom(data, train_idx, test_idx)
    data['room_diff'] = data['num_bathroom'] - data['num_bedroom']
    data = preprocessFips(data)
    data = preprocessRegion(data, train_idx, test_idx)
    data = preprocessLocation(data, train_idx, test_idx)
    data = preprocessCensusBlock(data)
    data = preprocessFinishedSquare(data)
    data = preprocessProperty(data)
    # 统计各值的分布，region_zip、property_zoing_encoder、region_neighbor放入效果不佳
    for col in ['room_diff', 'property_landuse']:
        print(col)
        data = distributionStatistics(data, col, train_idx, test_idx)
        data[col] = data[col].astype(np.float32)
    data = preprocessNumRoom(data)
    data = preprocessNumUnit(data)
    data = preprocessDeckType(data)
    data = preprocessStory(data)
    data = preprocessFirePlace(data)
    data = preprocessGarage(data)
    data = preprocessPool(data)
    data = preprocessYard(data)
    data = preprocessHeatingType(data)
    data = preprocessTax(data)
    data = preprocessBuiltYear(data)
    del data['parcel_id']
    data = preprocessBuilding(data)
    data = crossFeats(data, train_idx, test_idx)
    #data = preprocessParcelId(data)
    # 缺失值填充0
    data.fillna(0, inplace=True)

    # 筛选特征
    feats_in_use = [col for col in data.columns if col not in feats_not_use]

    data_train = np.array(data.loc[train_idx, :][feats_in_use])
    y_train = data.loc[train_idx, 'fips']
    data_test = np.array(data.loc[test_idx, :][feats_in_use])

    del data; gc.collect()

    stda = StandardScaler()
    data_train = stda.fit_transform(data_train)
    data_test = stda.transform(data_test)

    # 高基数特征处理，property_zoing_encoder、
    high_cardinality_feats = ['region_zip', 'region_neighbor','room_diff', 'raw_census_tract_and_block_encoder', 'property_landuse']
    # 该列在数据中的索引位置
    feats_idx = [feats_in_use.index(feat) for feat in high_cardinality_feats]
    # 对高基数特征进行处理
    woe_train, woe_test = convert_to_avg(data_train[:, feats_idx], y_train, data_test[:, feats_idx], seed=1, cvals=5, roundings=2, columns=None)
    print('preprocessHighCardinality: finished')

    # 合并特征
    data_train = np.hstack((data_train, woe_train))
    del woe_train; gc.collect()

    data_test = np.hstack((data_test, woe_test))
    del woe_test; gc.collect()

    for col in high_cardinality_feats:
        feats_in_use.append(col + '_cardinality')

    return data_train, data_test, feats_in_use

