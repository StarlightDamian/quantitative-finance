# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
from datetime import datetime
import argparse

import pandas as pd
import lightgbm as lgb

from __init__ import path
from base import base_connect_database
from train import train_main

#MODEL_PATH = f'{path}/checkpoint/prediction_stock_model.txt'
TEST_TABLE_NAME = 'prediction_stock_price_test'

class StockPredictionModel(train_main.StockTrainModel):
    def __init__(self, date_start, date_end):
        """
        Initialize StockPredictionModel object, including feature engineering and database connection.
        """
        super().__init__(date_start, date_end)
        
    def load_model(self, model_path):
        """
        Load model from the specified path.
        """
        self.model = lgb.Booster(model_file=model_path)

    def load_dataset(self, date_range_bunch):
        x, y = date_range_bunch.data, date_range_bunch.target
        x_test = pd.DataFrame(x, columns=date_range_bunch.feature_names)
        y_test = pd.DataFrame(y, columns=date_range_bunch.target_names)
        y_test = y_test.values.flatten()
        return None, x_test, None, y_test

    def data_processing_prediction(self, date_range_data):
        _, x_test, _, y_high_test, _, y_low_test,_ ,y_diff_test = self.feature_engineering_pipline(date_range_data)
        
        # 训练集的主键删除，测试集的主键抛出
        primary_key_test = x_test.pop('primaryKey')
        
        #stock_model.train_model(x_train, y_high_train)
        #self.load_model(MODEL_PATH)
        self.load_model(f'{path}/checkpoint/prediction_stock_high_model.txt')
        feature_names = self.model.dump_model()['feature_names']
        x_test = x_test[feature_names]
        y_high = self.prediction_y(y_high_test, x_test, task_name='test', prediction_name='high')
        y_high = y_high.rename(columns={0: 'rearHighPctChgReal',
                                        1: 'rearHighPctChgPred'})
        
        self.load_model(f'{path}/checkpoint/prediction_stock_low_model.txt')
        feature_names = self.model.dump_model()['feature_names']
        x_test = x_test[feature_names]
        y_low = self.prediction_y(y_low_test, x_test, task_name='test', prediction_name='low')
        y_low = y_low.rename(columns={0: 'rearLowPctChgReal',
                                      1: 'rearLowPctChgPred'})

        self.load_model(f'{path}/checkpoint/prediction_stock_diff_model.txt')
        feature_names = self.model.dump_model()['feature_names']
        x_test = x_test[feature_names]
        y_diff = self.prediction_y(y_diff_test, x_test, task_name='test', prediction_name='diff')
        y_diff = y_diff.rename(columns={0: 'rearDiffPctChgReal',
                                        1: 'rearDiffPctChgPred'})

        x_test = x_test.reset_index(drop=True)  # train_test_split过程中是保留了index的，在这一步重置index
        prediction_stock_price  = pd.concat([y_high, y_low, y_diff, x_test], axis=1)
        
        prediction_stock_price['remarks'] = prediction_stock_price.apply(lambda row: 'limit_up' if row['high'] == row['low'] else '', axis=1)
        prediction_stock_price = prediction_stock_price[['rearLowPctChgReal', 'rearLowPctChgPred', 'rearHighPctChgReal','rearHighPctChgPred', 
                                                         'rearDiffPctChgReal', 'rearDiffPctChgPred', 'open','high', 'low', 'close','volume',
                                                         'amount','turn', 'pctChg', 'remarks']]
                                                         
        # 通过主键关联字段
        related_name = ['date', 'code', 'code_name', 'isST']
        prediction_stock_price['primaryKey'] = primary_key_test
        prediction_stock_price_related = pd.merge(prediction_stock_price, date_range_data[['primaryKey']+related_name],on='primaryKey')
        
        with base_connect_database.engine_conn('postgre') as conn:
            prediction_stock_price_related['insert_timestamp'] = datetime.now().strftime('%F %T')
            prediction_stock_price_related.to_sql(TEST_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')  # 输出到数据库
            
        return prediction_stock_price_related
        
# =============================================================================
#     def data_prediction_pipline(self, date_range_data):
#         # 选择推荐模型
#         self.load_model(MODEL_PATH)
#         
#         prediction_stock_price, prediction_stock_price_related = self.data_processing_prediction(date_range_data)
#         return prediction_stock_price, prediction_stock_price_related
# =============================================================================
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-03-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-12-01', help='进行回测的结束时间')
    args = parser.parse_args()

    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        backtest_df = pd.read_sql(f"SELECT * FROM history_a_stock_k_data WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    print(backtest_df)
    
    stock_prediction_model = StockPredictionModel(date_start=args.date_start, date_end=args.date_end)
    prediction_stock_price_related = stock_prediction_model.data_processing_prediction(backtest_df)
    
    
    # path='E:/03_software_engineering/github/quantitative-finance/checkpoint'
    # model = lgb.Booster(model_file=f'{path}/prediction_stock_diff_model.txt')
    # model.dump_model()['feature_names']