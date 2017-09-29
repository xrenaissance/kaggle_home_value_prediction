#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from process import coreProcess
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
import gc
import os

###############################################################################################################
# 特征工程
def featDf(data, train_idx, test_idx, output_path):
    train_id = data.loc[train_idx, 'parcel_id']
    test_id = data.loc[test_idx, 'parcel_id']
    train_id.to_csv(output_path + 'train_id.csv')
    test_id.to_csv(output_path + 'test_id.csv')

    # 特征工程
    X_train, X_test, data_col = coreProcess(data, train_idx, test_idx)
    X_train = pd.DataFrame(X_train, columns=data_col)
    X_test = pd.DataFrame(X_test, columns=data_col)

    return X_train, X_test, train_id, test_id

###############################################################################################################
# 运行xgb1
def XGB1(X_train, X_test, y, sample, train_id, test_id):
    # xgboost参数设置
    y_mean = np.mean(y)
    xgb_params = {'eta': 0.002,
                  'max_depth': 5,
                  'subsample': 0.75,
                  'objective': 'reg:linear',
                  'eval_metric': 'mae',
                  'base_score': y_mean,
                  'silent': 1}

    d_train = xgb.DMatrix(X_train, y)
    d_test = xgb.DMatrix(X_test)

    # 交叉验证，train-mae:0.067744  test-mae:0.0682922，kaggle-0.0649577
    cv_result = xgb.cv(xgb_params, d_train, nfold=5, num_boost_round=1000,
                       early_stopping_rounds=400, verbose_eval=10, show_stdv=False)
    num_boost_rounds = len(cv_result)
    print(num_boost_rounds)

    # 训练模型
    model = xgb.train(dict(xgb_params, silent=1), d_train, num_boost_round=num_boost_rounds)
    pred_test = model.predict(d_test)
    pred_train = model.predict(d_train)
    del d_test, d_train; gc.collect()

    pred_test = pd.DataFrame(pred_test, columns=['predict'])
    pred_test['parcelid'] = list(test_id)
    pred_train = pd.DataFrame(pred_train, columns=['predict'])
    pred_train['parcelid'] = list(train_id)

    pred_data = pd.concat([pred_train, pred_test], axis=0)
    pred_data.drop_duplicates(['parcelid'], inplace=True)

    return pred_train, pred_test

###############################################################################################################
# 运行xgb2
def XGB2(X_train, X_test, y, sample, train_id, test_id, output_path):
    split = 80000
    x_train, y_train, x_valid, y_valid = X_train[:split], y[:split], X_train[split:], y[split:]

    y_mean = np.mean(y_train)
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    del x_train, x_valid; gc.collect()

    print('Training ...')
    params = {'eta': 0.002,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': 1,
              'subsample': 0.75,
              'base_score': y_mean}

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 20000, watchlist, early_stopping_rounds=400, verbose_eval=10)

    pred_train = clf.predict(d_train)
    pred_train = pd.DataFrame(pred_train, columns=['predict'])
    pred_train['parcelid'] = list(train_id)[:split]

    pred_valid = clf.predict(d_valid)
    pred_valid = pd.DataFrame(pred_valid, columns=['predict'])
    pred_valid['parcelid'] = list(train_id)[split:]
    pred_train = pd.concat([pred_train, pred_valid], axis=0)

    d_test = xgb.DMatrix(X_test)
    pred_test = clf.predict(d_test)
    pred_test = pd.DataFrame(pred_test, columns=['predict'])
    pred_test['parcelid'] = list(test_id)
    del d_test; gc.collect()

    pred_data = pd.concat([pred_train, pred_test], axis=0)
    pred_data.drop_duplicates(['parcelid'], inplace=True)

    # 特征重要性
    feat_df = pd.DataFrame(clf.get_fscore(), index=[0]).T
    feat_df = feat_df.reset_index()
    feat_df.columns = ['feat', 'fscore']
    feat_df.sort_values(by='fscore', inplace=True)

    gain_df = pd.DataFrame(clf.get_score(importance_type='gain'), index=[0]).T
    gain_df = gain_df.reset_index()
    gain_df.columns = ['feat', 'gain']

    feat_df = feat_df.merge(gain_df, on='feat')
    feat_df.to_csv(output_path + 'xgb2_feat_df.csv', index=None)

    return pred_data

###############################################################################################################
# 运行LGB
def LGB(X_train, X_test, y, sample, train_id, test_id):
    split = 8000
    x_train, y_train, x_valid, y_valid = X_train[:split], y[:split], X_train[split:], y[split:]
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)

    params = {'learning_rate': 0.002,
              'boosting_type': 'gbdt',
              'objective': 'regression',
              'metric': 'mae',
              'sub_feature': 0.5,
              'num_leaves': 60,
              'min_data': 500,
              'min_hessian': 1}
    # valid_0's l1: 0.0527277
    watchlist = [d_valid]
    clf = lgb.train(params, d_train, 900, watchlist)

    del d_train, d_valid, x_train , x_valid; gc.collect()

    clf.reset_parameter({"num_threads":1})
    pred_train = clf.predict(X_train)
    pred_train = pd.DataFrame(pred_train, columns=['predict'])
    pred_train['parcelid'] = list(train_id)

    pred_test = clf.predict(X_test)
    pred_test = pd.DataFrame(pred_test, columns=['predict'])
    pred_test['parcelid'] = list(test_id)

    pred_data = pd.concat([pred_train, pred_test], axis=0)
    pred_data.drop_duplicates(['parcelid'], inplace=True)

    # 特征重要性
    feat_df = pd.DataFrame(clf.feature_name(), columns=['lgb_feat'])
    feat_df['lgb_import'] = clf.feature_importance()
    feat_df.sort_values(by='lgb_import', inplace=True, ascending=False)
    feat_df.to_csv(output_path +'lgb_feat_df.csv', index=None)

    return pred_data


###############################################################################################################
# 参数搜索
def paramSearch(data, y, clf, tuned_params):
    best_clf = GridSearchCV(clf, tuned_params, scoring='neg_mean_absolute_error', n_jobs=1, cv=5)
    best_clf.fit(data, y)
    print(best_clf.best_params_)


###############################################################################################################
# 输出预测文件
def genSub(pred_data, sample, output_path):
    sub = sample.merge(pred_data, on='parcelid', how='left')
    for col in ['201610', '201611', '201612', '201710', '201711', '201712']:
        sub[col] = sub['predict']

    sub.drop(['predict'], axis=1, inplace=True)
    sub.to_csv(output_path + 'sub_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
               , index=False, float_format='%.4f')


if __name__ == '__main__':
    os.chdir(r'D:\PycharmProjects\kaggle_home_value_predictionn/')
     # 读取数据
    prop = pd.read_csv('properties_2016.csv')
    train = pd.read_csv('train_2016_v2.csv')
    sample = pd.read_csv('sample_submission.csv')

    sample['parcelid'] = sample['ParcelId']
    sample.drop(['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712'], axis=1, inplace=True)
    # 合并文件
    data = prop.merge(train, how='left', on='parcelid')
    # 重命名列
    col_dict = {'parcelid': 'parcel_id', 'logerror': 'log_error', 'transactiondate': 'transaction_date',
                'airconditioningtypeid': 'air_conditioning_type', 'basementsqft': 'area_basement',
                'bathroomcnt': 'num_bathroom', 'bedroomcnt': 'num_bedroom', 'buildingclasstypeid': 'building_class_type_id',
                'calculatedbathnbr': 'num_bathroom_calc', 'decktypeid': 'deck_type',
                'finishedfloor1squarefeet': 'area_firstfloor_finished', 'calculatedfinishedsquarefeet': 'area_total_calc',
                'finishedsquarefeet12': 'area_live_finished', 'finishedsquarefeet13': 'area_liveperi_finished',
                'finishedsquarefeet15': 'area_total_finished', 'finishedsquarefeet50': 'area_firstfloor',
                'finishedsquarefeet6': 'area_base', 'fips': 'fips', 'fireplacecnt': 'num_fireplace',
                'fullbathcnt': 'num_bath', 'garagecarcnt': 'num_garage', 'garagetotalsqft': 'area_garage',
                'hashottuborspa': 'flag_tub', 'heatingorsystemtypeid': 'heating_type', 'latitude': 'latitude',
                'longitude': 'longitude', 'lotsizesquarefeet': 'area_lot', 'poolcnt': 'num_pool',
                'poolsizesum': 'area_pool', 'pooltypeid10': 'pool_type_10', 'pooltypeid2': 'pool_type_2',
                'pooltypeid7': 'pool_type_7', 'propertycountylandusecode': 'property_county',
                'propertylandusetypeid': 'property_landuse', 'propertyzoningdesc': 'property_zoing',
                'rawcensustractandblock': 'raw_census_tract_and_block', 'regionidcity': 'region_city',
                'regionidcounty': 'region_county', 'regionidneighborhood': 'region_neighbor',
                'regionidzip': 'region_zip', 'roomcnt': 'num_room', 'storytypeid': 'story_type',
                'threequarterbathnbr': 'num_75_bath', 'typeconstructiontypeid': 'building_type',
                'unitcnt': 'num_unit', 'yardbuildingsqft17': 'area_patio', 'yardbuildingsqft26': 'area_shed',
                'yearbuilt': 'built_year', 'numberofstories': 'num_story', 'fireplaceflag': 'flag_fireplace',
                'structuretaxvaluedollarcnt': 'tax_structure', 'assessmentyear': 'assessment_year',
                'landtaxvaluedollarcnt': 'tax_land', 'taxamount': 'tax_property', 'taxdelinquencyflag': 'flag_tax_delinquency',
                'taxdelinquencyyear': 'tax_delinquency_year', 'censustractandblock': 'census_tract_and_block',
                'buildingqualitytypeid': 'building_quality_type', 'taxvaluedollarcnt': 'tax_value_num',
                'architecturalstyletypeid': 'building_style'}
    data.rename(columns=col_dict, inplace=True)
    # 训练集和测试集的索引
    train_idx = data[data['log_error'].notnull()].index
    test_idx = data[data['log_error'].isnull()].index

    # y值离群点设置
    # https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463/code
    #train_data = train_data[train_data['log_error'] > -0.4]
    #train_data = train_data[train_data['log_error'] < 0.4]
    #train_idx = train_data.index

    y = data.loc[data['log_error'].notnull(), 'log_error']
	y.index = range(len(y))
    data.drop(['log_error'], axis=1, inplace=True)

    output_path = r'D:\PycharmProjects\kaggle_home_value_prediction\result/'

    X_train, X_test, train_id, test_id = featDf(data, train_idx, test_idx, output_path)
    pred_data_xgb1 = XGB1(X_train, X_test, y, sample, train_id, test_id)
    pred_data_xgb2 = XGB2(X_train, X_test, y, sample, train_id, test_id, output_path)
    pred_data_lgb = LGB(X_train, X_test, y, sample, train_id, test_id)

    # 生成预测文件
    genSub(pred_data_lgb, sample, output_path)

    ###############################################################################################################
    # 参数搜索
    tuned_params = {'max_depth': [4, 5, 6, 7],
                'subsample': [0.4, 0.6, 0.8],
                'reg_lambda': [0.4, 0.6, 0.8],
                'reg_alpha': [0, 0.2, 0.4, 0.6, 0.8],
                'colsample_bytree': [0.4, 0.6, 0.8]}
    clf = xgb.XGBRegressor(learning_rate=0.02, n_estimators=1000, objective='reg:linear', gamma=0,
                       min_child_weight=1, max_delta_step=0, colsample_bylevel=1,
                       scale_pos_weight=1, base_score=0.5, seed=0)

    paramSearch(X_train, y, clf, tuned_params)
