from datetime import date,datetime
import numpy as np
import pandas as pd
import xgboost as xgb

import warnings


#计算用户收到优惠券和使用优惠券消费的时间间隔
def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8])) - date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days

#对特征数据集提取用户线下相关特征
def get_user_feature(df):
  #数量相关
  user = df[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
  u = user[['user_id']].copy().drop_duplicates()
  # 用户领取优惠券次数
  u1 = user[user['date_received'] != 'null'][['user_id']].copy()
  u1['coupon_received'] = 1
  u1 = u1.groupby(['user_id'], as_index=False).count()
  # 用户消费的次数
  u2 = user[user['date'] != 'null'][['user_id']].copy()
  u2['buy_total'] = 1
  u2 = u2.groupby(['user_id'], as_index=False).count()
  # 用户使用优惠券进行消费的次数
  u3 = user[((user['date'] != 'null') & (user['date_received'] != 'null'))][['user_id']].copy()
  u3['buy_use_coupon'] = 1
  u3 = u3.groupby(['user_id'], as_index=False).count()
  # 用户消费过的不同商家数量
  u4 = user[user['date'] != 'null'][['user_id', 'merchant_id']].copy()
  u4.drop_duplicates(inplace=True)
  u4 = u4.groupby(['user_id'], as_index=False).count()
  u4.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
  # 距离相关
  utmp = user[(user['date'] != 'null') & (user['date_received'] != 'null')][['user_id', 'distance']].copy()
  utmp.replace('null', -1, inplace=True)
  utmp.distance = utmp.distance.astype('int')
  utmp.replace(-1, np.nan, inplace=True)
  # 最小距离
  u5 = utmp.groupby(['user_id'], as_index=False).min()
  u5.rename(columns={'distance': 'user_min_distance'}, inplace=True)
  # 最大距离
  u6 = utmp.groupby(['user_id'], as_index=False).max()
  u6.rename(columns={'distance': 'user_max_distance'}, inplace=True)
  # 平均距离
  u7 = utmp.groupby(['user_id'], as_index=False).mean()
  u7.rename(columns={'distance': 'user_mean_distance'}, inplace=True)
  # 中位数距离
  u8 = utmp.groupby(['user_id'], as_index=False).median()
  u8.rename(columns={'distance': 'user_median_distance'}, inplace=True)
  # 时间间隔相关
  ugap = user[(user['date_received'] != 'null') & (user['date'] != 'null')][['user_id', 'date_received', 'date']]
  ugap['user_date_datereceived_gap'] = ugap['date'] + ':' + ugap['date_received']
  ugap['user_date_datereceived_gap'] = ugap['user_date_datereceived_gap'].apply(get_user_date_datereceived_gap)
  ugap = ugap[['user_id', 'user_date_datereceived_gap']]
  u9 = ugap.groupby(['user_id'], as_index=False).mean()
  u9.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
  u10 = ugap.groupby(['user_id'], as_index=False).min()
  u10.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
  u11 = ugap.groupby(['user_id'], as_index=False).max()
  u11.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)
  # 合并特征
  user_feature = pd.merge(u, u1, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u2, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u3, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u4, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u5, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u6, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u7, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u8, on='user_id', how='left')
  user_feature = pd.merge(user_feature, u9, on='user_id', how='left')
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

#读取数据
off_train = pd.read_csv(r'C:\Users\25125\Desktop\o2o数据集\ccf_offline_stage1_train.csv')
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

off_test = pd.read_csv(r'C:\Users\25125\Desktop\o2o数据集\ccf_offline_stage1_test_revised.csv')
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

on_train = pd.read_csv(r'C:\Users\25125\Desktop\o2o数据集\ccf_online_stage1_train.csv')
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


off_train.fillna(-1)
off_test.fillna(-1)
on_train.fillna(-1)
off_train = off_train.apply(pd.to_numeric,errors='ignore')
off_test = off_test.apply(pd.to_numeric,errors='ignore')
on_train = on_train.apply(pd.to_numeric,errors='ignore')

dataset1 = off_train[(off_train['date_received']>=20160414)&(off_train['date_received']<=20160514)]
dataset2 = off_train[(off_train['date_received']>=20160515)&(off_train['date_received']<=20160615)]
dataset3 = off_test
off_train.info()
#特征区间
feature1 = off_train[(off_train.date>=20160101)&(off_train.date<=20160413)|((off_train.date==0)&(off_train.date_received>=20160101)&(off_train.date_received<=20160413))]
feature2 = off_train[(off_train.date>=20160201)&(off_train.date<=20160514)|((off_train.date==0)&(off_train.date_received>=20160201)&(off_train.date_received<=20160514))]
feature3 = off_train[((off_train.date>=20160315)&(off_train.date<=20160630))|((off_train.date==0)&(off_train.date_received>=20160315)&(off_train.date_received<=20160630))]

print('Read Data Complete')
#对feature1提取用户线下相关特征
user1_feature = get_user_feature(feature1)
#转为csv格式并保存
user1_feature.to_csv(r'C:\Users\25125\Desktop\o2o数据集\user1_feature.csv',index=None)

#对feature2提取用户线下相关特征
user2_feature = get_user_feature(feature2)
#转为csv格式并保存
user2_feature.to_csv(r'C:\Users\25125\Desktop\o2o数据集\user2_feature.csv',index=None)

#对feature1提取用户线下相关特征
user3_feature = get_user_feature(feature3)
#转为csv格式并保存
user3_feature.to_csv(r'C:\Users\25125\Desktop\o2o数据集\user3_feature.csv',index=None)