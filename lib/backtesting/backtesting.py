# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
#path = 'C:/Users/awei/Desktop/github/quantitative-finance'
#import sys
#sys.path.append(path)
# Official Library
import argparse

# Third party libraries
import pandas as pd
import base_utils

def get_data_day(date_start, date_end):
    ...
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-01-01')
    parser.add_argument('--date_end', type=str, default='2023-03-01')
    args = parser.parse_args()
    
    
    #get_data_day(date_start=args.date_start, date_end=args.date_end)
    
    
    print(base_utils.date_binary_list('2022-04-03', '2022-04-07'))
    
    
    