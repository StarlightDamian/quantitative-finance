# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:40:07 2023

@author: awei
数据处理
"""
import pandas as pd

def re_get_row_data(rs):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    return result
