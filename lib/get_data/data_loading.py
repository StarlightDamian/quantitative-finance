# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:40:07 2023

@author: awei
数据处理
"""
import os
import argparse
import baostock as bs
from datetime import datetime

import numpy as np
import pandas as pd


from __init__ import path
from base import base_utils
import data_plate

def re_get_row_data(rs):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result

def feather_file_merge(date_start, date_end):
    date_binary_pair_list = base_utils.date_binary_list(date_start, date_end)
    feather_files = [f'{path}/data/day/{date_binary_pair[0]}.feather' for date_binary_pair in date_binary_pair_list]
    #print(feather_files)
    dfs = [pd.read_feather(file) for file in feather_files if os.path.exists(file)]
    feather_df = pd.concat(dfs, ignore_index=True)
    return feather_df


class PerformFeatureEngineering:
    def __init__(self):
        ...
    def feature_engineering_pipline():
        ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-02-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    
    backtest_df = feather_file_merge(args.date_start, args.date_end)
    print(backtest_df)
    
    # 新增日期差异作为日期特征
    #backtest_df['targetDate'] = datetime.now().strftime('%F') #应当以被测日期作为日差统计日期
    #backtest_df['dateDiff'] = (pd.to_datetime(backtest_df.targetDate) - pd.to_datetime(backtest_df.date)).dt.days
    
    
    # test = backtest_df[backtest_df.code == 'sz.399997']
    
    # 训练特征
    feature_names = ['code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST']#'dateDiff', 
    feature_df = backtest_df[feature_names]

    # 交易日,及其前置日期
    lg = bs.login()  # 登陆系统
    trading_day_df = data_plate.get_base_data(data_type='交易日')
    bs.logout()  # 登出系统
    trading_day_df = trading_day_df[trading_day_df.is_trading_day=='1']
    trading_day_df['pre_date'] = np.insert(trading_day_df.calendar_date, 0, '')[:-1]    
    #trading_day_df['rear_date'] = np.insert(trading_day_df.calendar_date, -1, '')[1:]    

    trading_day_pre_dict = trading_day_df.set_index('calendar_date')['pre_date'].to_dict()
    backtest_df['preDate'] = backtest_df.date.map(trading_day_pre_dict)
    
    #trading_day_rear_dict = trading_day_df.set_index('calendar_date')['rear_date'].to_dict()
    #backtest_df['rearDate'] = backtest_df.date.map(trading_day_rear_dict)
    backtest_df['primaryKey'] = backtest_df.date + backtest_df.code
    
    predict_pd = backtest_df[['preDate', 'code', 'high', 'low']]
    predict_pd.loc[:, ['primaryKey']] = predict_pd.preDate + predict_pd.code

    
    predict_pd = predict_pd.rename(columns={'high': 'rearHigh',
                                            'low': 'rearLow'})
    predict_pd = predict_pd[['primaryKey', 'rearHigh', 'rearLow']]
    
    # 关联对应后置最低最高价格
    backtest_df = pd.merge(backtest_df, predict_pd, on='primaryKey')
    
    # print(backtest_df[backtest_df.code=='sz.399997'][['date','rearDate','high','rearHigh']]) #观察数据
    
    backtest_df['rearHigh']
    backtest_df['rearLow']
    
    target_names = ['rearHigh', 'rearLow']
    target_df = backtest_df[target_names]
    
    backtest_dict = {'data': feature_df.values,
                     'feature_names': feature_names,
                     'target': target_df.values,
                     'target_names': target_names,}
    

    
    
    

# =============================================================================
# def days_until(date_str):
#     # 将输入的日期字符串转换为 datetime 对象
#     target_date = datetime.strptime(date_str, '%Y-%m-%d')
#     
#     # 获取今天的日期
#     today = datetime.now()
# 
#     # 计算日期差
#     delta = target_date - today
# 
#     # 返回天数
#     return delta.days
# =============================================================================
# 示例
#date_to_check = '2023-01-31'
#days_remaining = days_until(date_to_check)
#print(f"距离 {date_to_check} 还有 {days_remaining} 天")