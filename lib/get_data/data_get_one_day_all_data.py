# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:00:15 2023

@author: awei
获取指定日期全部股票的日K线数据
code_name 不属于特征，在这一层加入
"""
import argparse
from datetime import datetime
from sqlalchemy import Float, Numeric, String

import baostock as bs
import pandas as pd

from __init__ import path
from base import base_connect_database, base_utils

class GetDayData:
    def __init__(self, date_start="1990-01-01", date_end="3024-01-01"):
        bs.login()
        self.stock_df = pd.read_csv(f'{path}/data/all_stock.csv', encoding='gb18030')#.head(20)# 获取指数、股票数据
        
        self.date_start = date_start
        self.date_end = date_end
        
    def download_data_1(self, substring_pd):
        code = substring_pd.name
        # time = datetime.now().strftime('%F %T')
        # print(f'{code} {time}')
        print(f'date_start: {self.date_start}')
        print(f'date_end: {self.date_end}')
        k_rs = bs.query_history_k_data_plus(code,
                                            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
                                            self.date_start,
                                            self.date_end,
                                            )
        data_df = k_rs.get_data()
        
        # primaryKey主键不参与训练，用于关联对应数据
        data_df['primary_key'] = (data_df['date']+data_df['code']).apply(base_utils.md5_str) # md5（日期、时间、代码）
        return data_df
    
    def download_data(self):
        data_df = self.stock_df.groupby('code').apply(self.download_data_1)
        bs.logout()
        return data_df.reset_index(drop=True) 
    
    def data_handle(self, data_raw_df):
        data_raw_df = pd.merge(data_raw_df, self.stock_df[['code', 'code_name']], on='code')
        
        # 对异常值补全. 部分'amount'、'volume'为''
        columns_float_list = ['open', 'high', 'low', 'close', 'preclose']
        data_raw_df[columns_float_list] = data_raw_df[columns_float_list].fillna(0).astype(float)
        
        # 更高级别的异常处理
        columns_apply_float_list = ['amount', 'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']  # 选择需要处理的字段列表
        data_raw_df.loc[:, columns_apply_float_list] = data_raw_df[columns_apply_float_list].apply(pd.to_numeric, errors='coerce')  # 使用apply函数对每个字段进行处理
        data_raw_df[columns_apply_float_list] = data_raw_df[columns_apply_float_list].fillna(0).astype(float)  # 将 NaN 值填充为 0 或其他合适的值
        
        # volume中有异常值,太长无法使用.astype(int)。'adjustflag', 'tradestatus', 'isST',保持str
        data_raw_df.loc[:, ['volume']] = pd.to_numeric(data_raw_df['volume'], errors='coerce', downcast='integer')  # 使用pd.to_numeric进行转换，将错误的值替换为 NaN
        data_raw_df[['volume']] = data_raw_df[['volume']].fillna(0).astype('int64')  # 将 NaN 值填充为 0 或其他合适的值
        
        return data_raw_df[['primary_key', 'date', 'code', 'code_name', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'isST']]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2022-03-01', help='Start time for backtesting')#1990-01-01
    #parser.add_argument('--date_end', type=str, default='2023-06-01', help='End time for backtesting')
    args = parser.parse_args()
    
    
    try:
        with base_connect_database.engine_conn('postgre') as conn:
            max_date = pd.read_sql("SELECT max(date) FROM history_a_stock_k_data", con=conn.engine)
            print(max_date)
            date_start = max_date.values[0][0]
    except:
        date_start=args.date_start
        
    get_day_data = GetDayData(date_start=date_start)
    k_data_raw_df = get_day_data.download_data()
    k_data_raw_df.to_feather(f'{path}/data/history_a_stock_k_data.feather') #原生数据
    #k_data_raw_df = pd.read_feather(f'{path}/data/history_a_stock_k_data.feather')
    data_handle_df = get_day_data.data_handle(k_data_raw_df)
    print(f'数据量：{data_handle_df.shape[0]}')
    
    conn = base_connect_database.engine_conn('postgre')
    data_handle_df.to_sql('history_a_stock_k_data', con=conn.engine, index=False, if_exists='replace',
                            dtype={'primary_key': String,
                                    'date': String,
                                    'code': String,
                                    'code_name': String,
                                    'open': Float,
                                    'high': Float,
                                    'low': Float,
                                    'close': Float,
                                    'preclose': Float,
                                    'volume': Numeric,
                                    'amount': Numeric,
                                    'adjustflag': String,
                                    'turn': Float,
                                    'tradestatus': String,
                                    'pctChg': Float,
                                    'peTTM': Float,
                                    'psTTM': Float,
                                    'pcfNcfTTM': Float,
                                    'pbMRQ': Float,
                                    'isST': String,
                                    })
    
# =============================================================================
# def download_data(date):
#     bs.login()
# 
#     # 获取指定日期的指数、股票数据
#     stock_rs = bs.query_all_stock(date)
#     stock_df = stock_rs.get_data()
#     data_df = pd.DataFrame()
#     for code in stock_df["code"]:
#         print("Downloading :" + code)
#         k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close", date, date)
#         data_df = data_df.append(k_rs.get_data())
#     bs.logout()
#     data_df.to_csv("E:/download/demo_assignDayData.csv", encoding="gbk", index=False)
#     print(data_df)
# 
# 
# if __name__ == '__main__':
#     # 获取指定日期全部股票的日K线数据
#     download_data("2019-02-25")
# 
# =============================================================================
