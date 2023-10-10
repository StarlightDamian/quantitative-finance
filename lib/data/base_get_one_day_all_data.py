# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:00:15 2023

@author: awei
指定日期所有股
"""
path = 'C:/Users/awei/Desktop/github/quantitative-finance'
import baostock as bs
import pandas as pd
from datetime import datetime

class GetDayData:
    def __init__(self):
        bs.login()
        self.stock_df = pd.read_csv(f'{path}/data/A_share_code.csv')#.head(20)# 获取指数、股票数据
        
    def download_data_1(self, substring_pd):
        code = substring_pd.code.values[0]
        time = datetime.now().strftime('%F %T')
        print(f'{code} {time}')
        k_rs = bs.query_history_k_data_plus(code,
                                            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                            "1990-01-01", "2024-01-01")
        return k_rs.get_data()
    
    def download_data(self):
        data_df = self.stock_df.groupby('code').apply(self.download_data_1)
        data_df = data_df.reset_index(drop=True) 
        output_df = pd.merge(data_df, self.stock_df[['code', 'code_name']], on='code')
        output_df = output_df[['date', 'code', 'code_name', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'isST']]
        bs.logout()
        return output_df
    
    
if __name__ == '__main__':
    get_day_data = GetDayData()
    # 获取指定日期全部股票的日K线数据
    output_df = get_day_data.download_data()
    #output_df.to_csv("E:/download/demo_assignDayData.csv", encoding="gbk", index=False)
    output_df.to_feather(f'{path}/data/k_day.feather')
    print(output_df)


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
