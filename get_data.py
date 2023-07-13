# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:57:51 2023

@author: awei
"""


import tushare as ts
ts.set_token('73b2d291e120d8438a4507fd998912fb9cf752bf3410d4f05d959caa')
#pro = ts.pro_api()
#pro = ts.pro_api('your token')


df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
#df = pro.query('trade_cal', exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')






