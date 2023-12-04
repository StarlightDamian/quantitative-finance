# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:07:09 2023
sparse
@author: awei
特征工程主程序
feature_engineering_main
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch
import baostock as bs

from __init__ import path
from get_data import data_loading, data_plate
from base import base_connect_database
pd.options.mode.chained_assignment = None

class PerformFeatureEngineering:
    def __init__(self):
        """
        初始化函数，用于登录系统和加载行业分类数据
        :param check:是否检修中间层date_range_data
        """
        try:
            conn_pg = base_connect_database.engine_conn('postgre')
        except Exception as e:
            print(f"数据库获取数据异常: {e}")
        
        trading_day_df = pd.read_sql('trade_datas', con=conn_pg.engine)
        self.trading_day_df = trading_day_df[trading_day_df.is_trading_day=='1']
        
        # 行业分类数据
        stock_industry = pd.read_sql('stock_industry', con=conn_pg.engine)
        #stock_industry.loc[stock_industry.industry.isnull(), 'industry'] = '其他' # 不能在这步补全，《行业分类数据》不够完整会导致industry为nan
        self.code_and_industry_dict = stock_industry.set_index('code')['industry'].to_dict()
        
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.target_names = ['rearHighPctChgPred', 'rearLowPctChgPred']
        
    def specified_trading_day(self, pre_date_num=1):
        """
        获取指定交易日的字典，用于预测日期的计算
        :param pre_date_num: 前置日期数，默认为1
        :return: 字典，包含指定交易日和对应的前置日期
        """
        trading_day_df = self.trading_day_df
        trading_day_df['pre_date'] = np.insert(trading_day_df.calendar_date, 0, ['']*pre_date_num)[:-pre_date_num]    
        trading_day_pre_dict = trading_day_df.set_index('calendar_date')['pre_date'].to_dict()
        return trading_day_pre_dict

    def create_values_to_predicted(self, date_range_data):
        """
        制作待预测值，为后一天的最高价和最低价
        :param date_range_data: 包含日期范围的DataFrame
        :return: 包含待预测值的DataFrame
        """
        date_range_data['primaryKey'] = date_range_data.date + date_range_data.code
        
        # 待预测的指定交易日的主键、价格
        predict_pd = date_range_data[['targetDate', 'code', 'high', 'low']]
        predict_pd.loc[:, ['primaryKey']] = predict_pd.targetDate + predict_pd.code
        predict_pd = predict_pd.rename(columns={'high': 'rearHigh',
                                                'low': 'rearLow'})
        predict_pd = predict_pd[['primaryKey', 'rearHigh', 'rearLow']]
        
        # 关联对应后置最低最高价格
        date_range_data = pd.merge(date_range_data, predict_pd, on='primaryKey')
        # print(date_range_data[date_range_data.code=='sz.399997'][['date','rearDate','high','rearHigh']]) #观察数据
        
        return date_range_data
    
    def build_features(self, date_range_data):
        """
        构建数据集，将DataFrame转换为Bunch
        :param date_range_data: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        ## 训练特征
        # 特征: 日期差异作为日期特征
        date_range_data['dateDiff'] = (pd.to_datetime(date_range_data.targetDate) - pd.to_datetime(date_range_data.date)).dt.days
        
        # 特征: 行业
        date_range_data['industry'] = date_range_data.code.map(self.code_and_industry_dict)
        date_range_data.loc[date_range_data.industry.isnull(), 'industry'] = '其他'
        
        # lightgbm不支持str格式，把行业类别转化为ont-hot
        one_hot_industry_array = self.one_hot_encoder.fit_transform(date_range_data[['industry']])
        one_hot_columns_list = self.one_hot_encoder.get_feature_names_out(['industry']).tolist()
        #print('one_hot_columns_list',one_hot_columns_list)
        one_hot_industry_df = pd.DataFrame(one_hot_industry_array, columns=one_hot_columns_list)
        date_range_data = pd.concat([date_range_data, one_hot_industry_df], axis=1)
        
        feature_int_columns = ['dateDiff', 'tradestatus', 'isST'] + one_hot_columns_list
        date_range_data.loc[date_range_data.isST=='', :] = 0 # 600万数据存在13万数据isST==''
        date_range_data[feature_int_columns] = date_range_data[feature_int_columns].astype(int)
        date_range_data[['open', 'high', 'low', 'close']] = date_range_data[['open', 'high', 'low', 'close']].astype(float)
        
        feature_names = ['dateDiff', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST']  + one_hot_columns_list
        feature_df = date_range_data[feature_names]
        
        # volume中有异常值
        feature_df.loc[:, ['volume']] = pd.to_numeric(feature_df['volume'], errors='coerce', downcast='integer')  # 使用pd.to_numeric进行转换，将错误的值替换为 NaN
        feature_df[['volume']] = feature_df[['volume']].fillna(0).astype('int64')  # 将 NaN 值填充为 0 或其他合适的值
        
        # amount、turn、pctChg中有''
        fields_to_convert = ['amount', 'turn', 'pctChg']  # 选择需要处理的字段列表
        feature_df.loc[:, fields_to_convert] = feature_df[fields_to_convert].apply(pd.to_numeric, errors='coerce')  # 使用apply函数对每个字段进行处理
        feature_df[fields_to_convert] = feature_df[fields_to_convert].fillna(0).astype(float)  # 将 NaN 值填充为 0 或其他合适的值
        
        date_range_data[['rearHigh', 'rearLow']] = date_range_data[['rearHigh', 'rearLow']].astype(float)
        
        # 明日最高值相对于今日收盘价的涨跌幅
        date_range_data['rearHighPctChgPred'] = ((date_range_data['rearHigh'] - date_range_data['close']) / date_range_data['close']) * 100
        date_range_data['rearLowPctChgPred'] = ((date_range_data['rearLow'] - date_range_data['close']) / date_range_data['close']) * 100
        
        target_df = date_range_data[self.target_names]  # 机器学习预测值
        return date_range_data, feature_df, target_df, feature_names
    
    def build_dataset(self, feature_df, target_df, feature_names):
        # 最高价格数据集
        date_range_high_dict = {'data': np.array(feature_df.to_records(index=False)),  # 不使用 feature_df.values,使用结构化数组保存每一列的类型
                         'feature_names': feature_names,
                         'target': target_df[self.target_names[0]].values,
                         'target_names': [self.target_names[0]],
                         }
        date_range_high_bunch = Bunch(**date_range_high_dict)
        
        # 最低价格数据集
        date_range_low_dict = {'data': np.array(feature_df.to_records(index=False)),
                         'feature_names': feature_names,
                         'target': target_df[self.target_names[1]].values,
                         'target_names': [self.target_names[1]],
                         }
        date_range_low_bunch = Bunch(**date_range_low_dict)
        return date_range_high_bunch, date_range_low_bunch
    
    def feature_engineering_pipline(self, date_range_data):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param date_range_data: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        trading_day_target_dict = self.specified_trading_day(pre_date_num=1)
        date_range_data['targetDate'] = date_range_data.date.map(trading_day_target_dict)
        date_range_data = self.create_values_to_predicted(date_range_data)
        
        # 构建数据集
        date_range_data, feature_df, target_df, feature_names = self.build_features(date_range_data)
        #date_range_high_bunch, date_range_low_bunch = self.build_dataset(feature_df, target_df, feature_names)
        return date_range_data, feature_df, target_df, feature_names
    
    def feature_engineering_dataset_pipline(self, date_range_data):
        """
        特征工程的主要流程，包括指定交易日、创建待预测值、构建数据集
        :param date_range_data: 包含日期范围的DataFrame
        :return: 包含数据集的Bunch
        """
        # 构建数据集
        date_range_data, feature_df, target_df, feature_names= self.feature_engineering_pipline(date_range_data)
        date_range_high_bunch, date_range_low_bunch = self.build_dataset(feature_df, target_df, feature_names)
        return date_range_high_bunch, date_range_low_bunch
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2022-03-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-03-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    # 获取日期段数据
    date_range_data = data_loading.feather_file_merge(args.date_start, args.date_end)
    print(date_range_data)
    
    # 特征工程
    perform_feature_engineering = PerformFeatureEngineering()
    
    # 特征工程结果
    date_range_data, feature_df, target_df, feature_names = perform_feature_engineering.feature_engineering_pipline(date_range_data)
    
    #date_range_high_bunch, date_range_low_bunch = perform_feature_engineering.feature_engineering_pipline(date_range_data)
    #print(date_range_high_bunch, date_range_low_bunch)
    

