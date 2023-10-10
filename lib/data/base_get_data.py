# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:57:51 2023

@author: awei
指定股,指定时间段
"""
import baostock as bs

from base import base_data_loading
from __init__ import path

def get_one_day_data(stock_code, start_date, end_date):
    rs = bs.query_history_k_data_plus(stock_code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3") #frequency="d"取日k线，adjustflag="3"默认不复权
    
    result = base_data_loading.re_get_row_data(rs)
    return result



if __name__ == "__main__":
    lg = bs.login()
    
    result = get_one_day_data(stock_code="sh.600000", start_date="1980-06-01", end_date="2023-12-31")
    result.to_csv(f"{path}/data/history_k_data.csv", encoding="gbk", index=False)
    print(result)
    
    bs.logout()
    
    