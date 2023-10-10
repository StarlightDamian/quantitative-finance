# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
获取股票对应的板块
"""
import baostock as bs

from base import base_data_loading
from __init__ import path

# 登陆系统
lg = bs.login()

# 显示登陆返回信息
# print('login respond error_code:'+lg.error_code)
# print('login respond  error_msg:'+lg.error_msg)

# 获取行业分类数据
rs = bs.query_stock_industry()
# rs = bs.query_stock_basic(code_name="浦发银行")
#print('query_stock_industry error_code:'+rs.error_code)
#print('query_stock_industry respond  error_msg:'+rs.error_msg)

# 打印结果集
result = base_data_loading.re_get_row_data(rs)

# 结果集输出到csv文件
result.to_csv(f"{path}/data/stock_industry.csv", encoding="gbk", index=False)
print(result)

# 登出系统
bs.logout()

if __name__ == '__main__':
    ...
    
