# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:37:21 2018

@author: FNo0
"""

import pandas as pd
import xgboost as xgb
import warnings
import numpy as np

warnings.filterwarnings('ignore')  # 不显示警告


def prepareTimeData(data):
    # time prepare
    data['date_received'] = pd.to_datetime(data['Date_received'],format='%Y%m%d')
    if 'Date' in data.columns.tolist():
        data['date'] = pd.to_datetime(data['Date'],format='%Y%m%d')
    return data

def prepareDistanceData_model(data):
    # help prepare distance data
    if data == -1:
        return 1
    else:
        return 0

def prepareDistanceData(data):
    # distance prepare
    data['Distance'].fillna(-1, inplace=True)
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    return data



def prepareRateData_model1(data):
    # choice Type
    if ':' in str(data):
        return 1
    else:
        return 0


def prepareRateData_model2(data):
    # Type conversion
    if ':' in str(data):
        return float(data)
    else:
        module = (float(str(data).split(':')[0]) - float(str(data).split(':')[1])) / float(str(data).split(':')[0])
        return module


def prepareRateData_model1_exchange(data):
    if ':' in str(data):
        return -1
    else:
        mode = int(str(data).split(':')[0])
        return mode

def prepareRateData(data):
    # rate data prepare
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减全部转换为折扣率
    data['min_cost_of_manjian'] = data['Discount_rate'].map(lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))  # 满减的最低消费
    return data

def get_label_model(data,data_received):
    if (data-data_received).total_seconds() / (60*60*24) <= 15:
        return 1
    else:
        return 0

def get_label(data):
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    return data

def feature_usess(data):
    dataset = data.copy()
    d = dataset[['user_id']].copy().drop_duplicates()
    # 用户领取优惠券次数
    done = dataset[dataset['date_received'] != 'null'][['user_id']].copy()
    done['coupon_received'] = 1
    done = done.groupby(['user_id'], as_index=False).count()

    # 用户消费的次数

    dtwo = dataset[dataset['date'] != 'null'][['user_id']].copy()
    dtwo['buy_total'] = 1
    dtwo = dtwo.groupby(['user_id'], as_index=False).count()

    # 用户使用优惠券进行消费的次数
    dthree = dataset[((dataset['date'] != 'null') & (dataset['date_received'] != 'null'))][['user_id']].copy()
    dthree['buy_use_coupon'] = 1
    dthree = dthree.groupby(['user_id'], as_index=False).count()
    user_feature = pd.merge(d, done, on='user_id', how='left')

    user_feature = pd.merge(user_feature, dtwo, on='user_id', how='left')

    user_feature = pd.merge(user_feature, dthree, on='user_id', how='left')
    # 用户消费过的不同商家数量

    dfour = dataset[dataset['date'] != 'null'][['user_id', 'merchant_id']].copy()
    dfour.drop_duplicates(inplace=True)
    dfour = dfour.groupby(['user_id'], as_index=False).count()
    dfour.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
    # 距离相关\

    distance = dataset[(dataset['date'] != 'null') & (dataset['date_received'] != 'null')][['user_id', 'distance']].copy()
    distance.replace('null', -1, inplace=True)
    distance.distance = distance.distance.astype('int')
    distance.replace(-1, np.nan, inplace=True)
    # 最小距离


    dfive = distance.groupby(['user_id'], as_index=False).min()
    dfive.rename(columns={'distance': 'user_min_distance'}, inplace=True)
    user_feature = pd.merge(user_feature, dfour, on='user_id', how='left')

    user_feature = pd.merge(user_feature, dfive, on='user_id', how='left')
    # 最大距离


    dsix = distance.groupby(['user_id'], as_index=False).max()
    dsix.rename(columns={'distance': 'user_max_distance'}, inplace=True)
    # 平均距离

    u7 = distance.groupby(['user_id'], as_index=False).mean()
    u7.rename(columns={'distance': 'user_mean_distance'}, inplace=True)
    # 中位数距离

    u9 = user_feature
    dfa = distance.groupby(['user_id'], as_index=False).median()
    dfa.rename(columns={'distance': 'user_median_distance'}, inplace=True)
    user_feature = pd.merge(user_feature, dfa, on='user_id', how='left')


    user_feature = pd.merge(user_feature, u9, on='user_id', how='left')
    # 时间间隔相关
    ugap = dataset[(dataset['date_received'] != 'null') & (dataset['date'] != 'null')][['user_id', 'date_received', 'date']]
    ugap['user_date_datereceived_gap'] = ugap['date'] + ':' + ugap['date_received']
    ugap['user_date_datereceived_gap'] = ugap['user_date_datereceived_gap'].apply()
    ugap = ugap[['user_id', 'user_date_datereceived_gap']]
    u9 = ugap.groupby(['user_id'], as_index=False).mean()
    u9.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    u10 = ugap.groupby(['user_id'], as_index=False).min()
    user_feature = pd.merge(user_feature, dsix, on='user_id', how='left')


    user_feature = pd.merge(user_feature, u7, on='user_id', how='left')
    u10.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    u11 = ugap.groupby(['user_id'], as_index=False).max()
    u11.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)


    user_feature = pd.merge(user_feature, u10, on='user_id', how='left')

    user_feature = pd.merge(user_feature, u11, on='user_id', how='left')
    # 增加两个比例相关的特征
    user_feature['buy_use_coupon_rate'] = user_feature['buy_use_coupon'].astype('float') / user1_feature[
        'buy_total'].astype('float')
    user_feature['user_coupon_transfer_rate'] = user_feature['buy_use_coupon'].astype('float') / user1_feature[
        'coupon_received'].astype('float')
    # 将np.nan用0进行替代
    user_feature = user_feature.fillna(0)
    return user_feature

def changeType(data):
    data['Date_received'] = data['Date_received'].map(int)
    data['Distance'] = data['Distance'].map(int)
    data['User_id'] = data['User_id'].fillna(0)
    data['User_id'] = data['User_id'].map(int)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Coupon_id'] = data['Coupon_id'].fillna(0)
    data['Date_received'] = data['Date_received'].fillna(0)
    data['Distance'] = data['Distance'].fillna(0)

    if 'label' in data.columns.tolist():
        data['label'] = data['label'].fillna(0)
        data['label'] = data['label'].map(int)
    return data

def get_histore_feature(history_field):

    data = history_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1  # 方便特征提取
    feature = data.copy()
    keys = ['User_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': '用户领取优惠券次数'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': '用户未核销的次数'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': '用户核销次数'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.drop(['cnt'], axis=1, inplace=True)
    return feature


def get_simple_feature(label_field):

    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1
    feature = data.copy()
    keys = ['User_id', 'Coupon_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': 'one'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    keys = ['Coupon_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': 'two'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    keys = ['User_id', 'Merchant_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 0 if len(x) > 1 else 1)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': 'three'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    keys = ['User_id', 'Coupon_id']
    pivot = pd.pivot_table(data, index=keys, values='Date_received', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Date_received': 'wiff'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature['first'] = list(
        map(lambda x, y: 1 if str(x) == str(y) else 0, feature['Date_received'], feature['wiff']))
    keys = ['User_id', 'Coupon_id']
    pivot = pd.pivot_table(data, index=keys, values='Date_received', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Date_received': 'laf'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature['last'] = list(
        map(lambda x, y: 1 if str(x) == str(y) else 0, feature['Date_received'], feature['laf']))
    feature['weekday_Receive'] = feature['date_received'].apply(lambda x: x.isoweekday())
    feature['moonday_Receive'] = feature['Date_received'].apply(
        lambda x: float(str(x)[-4:-2]) if str(x) != 'nan' else 0)
    feature['双休日'] = feature['weekday_Receive'].apply(lambda x: 1 if x == 6 or 7 else 0)
    feature['月初'] = feature['moonday_Receive'].apply(lambda x: 1 if x == 1 else 0)


    feature.drop(['cnt'], axis=1, inplace=True)

    # 返回
    return feature


def model_1(da):
    if 'Date' in da.columns.tolist():  # 表示训练集和验证集
        da.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = da['label'].tolist()
        da.drop(['label'], axis=1, inplace=True)
        da['label'] = label
    else:  # 表示测试集
        da.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    return da


def get_dataset(history_field, label_field):

    # 特征工程
    simple_feat = get_simple_feature(label_field)  # 示例简单特征
    history_feat = get_histore_feature(history_field)
    # 构造数据集
    dataset = pd.merge(simple_feat, history_feat, how='left')
    dataset = model_1(dataset)
    dataset = changeType(dataset)
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    # 返回
    return dataset


def model_xgb(tr, te):

    tr = pd.DataFrame(tr)
    te = pd.DataFrame(te)
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1.1,
              'gamma': 0.1,
              'lambda': 10,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.7,
              'scale_pos_weight': 1}

    data_train = xgb.DMatrix(tr.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=tr['label'])
    data_test = xgb.DMatrix(te.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    list = [(data_train, 'train')]
    model = xgb.train(params, data_train, num_boost_round=500, evals=list)
    foresee = model.predict(data_test)
    foresee = pd.DataFrame(foresee, columns=['prob'])
    result = pd.concat([te[['User_id', 'Coupon_id', 'Date_received']], foresee], axis=1)
    return result


if __name__ == '__main__':
    # 源数据
    off_train = pd.read_csv(r'C:\Users\25125\Desktop\o2o数据集\ccf_offline_stage1_train.csv')
    off_test = pd.read_csv(r'C:\Users\25125\Desktop\o2o数据集\ccf_offline_stage1_test_revised.csv')
    # 预处理
    off_train = prepareTimeData(off_train)
    off_train = prepareDistanceData(off_train)
    off_train = prepareRateData(off_train)
    off_test = prepareTimeData(off_test)
    off_test = prepareDistanceData(off_test)
    off_test = prepareRateData(off_test)
    off_train = get_label(off_train)
    # 划分区间
    train_history = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_label = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、中间区间、标签区间
    validate_history = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_label = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、中间区间、标签区间
    test_history = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_label= off_test.copy()  # [20160701,20160801)
    # 构造训练集、验证集、测试集
    train = get_dataset(train_history, train_label)
    validate = get_dataset(validate_history, validate_label)
    test = get_dataset(test_history, test_label)
    test.drop(['label'], axis=1, inplace=True)
    # 线下验证
    # 线上训练
    big_train = pd.concat([train, validate], axis=0)
    result = model_xgb(big_train, test)
    # 保存
    result.to_csv(r'C:\Users\25125\Desktop\o2o数据集\easy.csv', index=False, header=None)