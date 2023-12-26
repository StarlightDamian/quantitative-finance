# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
from datetime import datetime
import argparse

import joblib
import pandas as pd

from __init__ import path
from base import base_connect_database
from train import train_main

TEST_TABLE_NAME = 'prediction_stock_price_test'
MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/prediction_stock_lgbm_model.joblib'


class StockPredictionModel(train_main.StockTrainModel):
    def __init__(self):#, date_start, date_end
        """
        Initialize StockPredictionModel object, including feature engineering and database connection.
        """
        super().__init__()#date_start, date_end
        
    def load_model(self, model_path):
        """
        Load model from the specified path.
        """
        self.model_multioutput_regressor, model_metadata = joblib.load(MULTIOUTPUT_MODEL_PATH)
        feature_names = model_metadata['feature_names']
        primary_key_name = model_metadata['primary_key_name']
        return feature_names, primary_key_name
        
    def load_dataset(self, date_range_bunch):
        x, y = date_range_bunch.data, date_range_bunch.target
        x_test = pd.DataFrame(x, columns=date_range_bunch.feature_names)
        y_test = pd.DataFrame(y, columns=date_range_bunch.target_names)
        return None, x_test, None, y_test
    
    
    def data_processing_prediction(self, date_range_data):
        #global date_range_data1
        #date_range_data1 = date_range_data
        _, x_test, _, y_test = self.feature_engineering_pipline(date_range_data)
        
        feature_names, primary_key_name = self.load_model(MULTIOUTPUT_MODEL_PATH)
        
        primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
        x_test = x_test.reindex(columns=feature_names, fill_value=False)  # Pop first, then reindex
        
        y_result = self.prediction_y(y_test, x_test, task_name='test')
        prediction_stock_price = self.field_handle(y_result, x_test)
        
        # 通过主键关联字段
        related_name = ['date', 'code', 'code_name', 'preclose', 'isST']  # partition_date
        prediction_stock_price['primary_key'] = primary_key_test
        prediction_stock_price_related = pd.merge(prediction_stock_price, date_range_data[['primary_key']+related_name], on='primary_key')
        
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
    parser.add_argument('--date_start', type=str, default='2020-01-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-01-01', help='End time for backtesting')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        backtest_df = pd.read_sql(f"SELECT * FROM history_a_stock_k_data WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    print(backtest_df)
    
    stock_prediction_model = StockPredictionModel()#date_start=args.date_start, date_end=args.date_end
    prediction_stock_price_related = stock_prediction_model.data_processing_prediction(backtest_df)
    
    
    # path='E:/03_software_engineering/github/quantitative-finance/checkpoint'
    # model = lgb.Booster(model_file=f'{path}/prediction_stock_diff_model.txt')
    # model.dump_model()['feature_names']
    
    
    # =============================================================================
    #         x_test = x_test[['industry_国防军工', 'open', 'industry_其他', 'high', 'close',
    #                'industry_休闲服务', 'industry_计算机', 'industry_建筑材料', 'industry_非银金融',
    #                'industry_公用事业', 'pctChg', 'industry_传媒', 'industry_化工',
    #                'industry_建筑装饰', 'industry_电子', 'industry_机械设备', 'isST_1',
    #                'industry_交通运输', 'industry_综合', 'turn', 'industry_房地产', 'preclose',
    #                'isST_0', 'pbMRQ', 'amount', 'dateDiff', 'industry_商业贸易',
    #                'industry_家用电器', 'industry_医药生物', 'industry_电气设备', 'industry_钢铁',
    #                'industry_轻工制造', 'tradestatus_1', 'industry_农林牧渔', 'industry_采掘',
    #                'peTTM', 'low', 'psTTM', 'primaryKey', 'industry_银行', 'industry_食品饮料',
    #                'industry_纺织服装', 'industry_通信', 'industry_有色金属', 'industry_汽车',
    #                'pcfNcfTTM', 'volume', 'tradestatus_0']]
    # =============================================================================