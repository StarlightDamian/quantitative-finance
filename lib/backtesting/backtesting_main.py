# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
import argparse

import pandas as pd

from __init__ import path
from get_data import data_loading  # , data_plate 
from feature_engineering import feature_engineering_main

def get_data_day(date_start, date_end):
    ...
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-02-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-03-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    #get_data_day(date_start=args.date_start, date_end=args.date_end)
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    # 获取日期段数据
    backtest_df = data_loading.feather_file_merge(args.date_start, args.date_end)
    print(backtest_df)
    
    # 特征工程
    perform_feature_engineering = feature_engineering_main.PerformFeatureEngineering()
    backtest_dict = perform_feature_engineering.feature_engineering_pipline(backtest_df)
    print(backtest_dict)
    
    
    x, y = backtest_dict.data, backtest_dict.target
    x_df = DataFrame(x, columns= backtest_dict.feature_names)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.15)
    
    