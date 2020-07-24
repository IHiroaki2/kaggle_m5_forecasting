#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:30:19 2020

@author: hiroaki_ikeshita
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import shutil
import gc
import os
import pdb
import math
import pickle
from sympy import *
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from scipy.sparse import csr_matrix
import sys,time, warnings, psutil, random
from multiprocessing import Pool
import decimal
import holidays
import json

warnings.filterwarnings('ignore')

from  myfunctions import reduce_mem_usage, seed_everything, weight_calc

CATEGORY_ID = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]  ####
category_name = "store"

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)


def total_weight():
    ### make WEIGHT_SCALED_42840, WEIGHT_FORMAT, WEIGHT_SCALED_30490
    
    train_val_df = pd.read_csv(jsn["TRAIN_DATA_PATH"])
    sell_price_df = pd.read_csv(jsn["SELL_PRICES_PATH"])
    calendar_df = pd.read_csv(jsn["CALENDAR_PATH"])
    
    day_list = []
    for i in range(1942, 1970):
        day_list.append("d_{}".format(i))
    for day in day_list:
        train_val_df[day] = np.nan
        
    melt_sales = pd.melt(train_val_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name="day", value_name="sales")
    calendar_df['date'] = pd.to_datetime(calendar_df['date'], format='%Y-%m-%d')
    melt_sales = melt_sales.merge(calendar_df, left_on="day", right_on="d")
    melt_sales = pd.merge(melt_sales, sell_price_df, on=["store_id", "item_id", "wm_yr_wk"], how='left')
    
    melt_sales = melt_sales[["id", "date", "sales", "sell_price"]]
    
    melt_sales = reduce_mem_usage(melt_sales)
    
    ############## Creat mat 
    # train_val_df = train_val_df[train_val_df["distribution"]=="poisson"]
    
    NUM_ITEMS = train_val_df.shape[0]
    
    product = train_val_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    
    # Creating weight mats
    weight_mat = np.c_[np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                       pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                       pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values,
                       np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
                       ].T
    
    np.save(jsn["WEIGHTS_DIR"] + "weight_mat_total.npy" ,weight_mat)
    weight_mat_csr = csr_matrix(weight_mat)
    
    #     del train_val_df, sell_price_df, calendar_df, weight_mat
    #     gc.collect()
    
    ############## Loss function weights are calculated and stored.
    weight1, weight2 = weight_calc(melt_sales, product, weight_mat_csr)
    # del df, train_val_df; gc.collect()
    #     del melt_sales; gc.collect()
    
    ############## SAVE WEIGHT
    
    np.save(jsn["WEIGHTS_DIR"] + "weight1_total.npy" ,weight1)
    np.save(jsn["WEIGHTS_DIR"] + "weight2_total.npy" ,weight2)
    
    # train_val_df = pd.read_csv('/Users/hiroaki_ikeshita/myfolder/diveintocode-ml/Kaggle/Walmart/m5-forecasting-accuracy/sales_train_evaluation.csv')
    # product = train_val_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    df = pd.DataFrame(np.ones([30490,1]).astype(np.int8), index=product.index, columns=["total"])
    df = pd.concat((df, pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8')), axis=1)
    df = pd.concat((df, pd.DataFrame(np.identity(30490).astype(np.int8), index=product.index, columns=product["id"])), axis=1)
    
    df.index = product.id
    df = df.T
    
    df.to_pickle(jsn["WEIGHTS_DIR"] + "weight_size_change_format.pkl")
    
    weight_scaled = weight2**2/weight1
    
    WEIGHT_SCALED_42840 = weight_scaled / weight_scaled[0]
    WEIGHT_SCALED_30490 = np.dot(WEIGHT_SCALED_42840, df.values)
    
    np.save(jsn["WEIGHTS_DIR"] + "WEIGHT_SCALED_42840.npy", WEIGHT_SCALED_42840)
    np.save(jsn["WEIGHTS_DIR"] + "WEIGHT_SCALED_30490.npy", WEIGHT_SCALED_30490)




def weight_by_store():
    ################################# Make WEIGHT
    #################################################################################
    
    ############## LOAD ORIGIN DATA
    
    train_val_df_origin = pd.read_csv(jsn["TRAIN_DATA_PATH"])
    sell_price_df = pd.read_csv(jsn["SELL_PRICES_PATH"])
    calendar_df = pd.read_csv(jsn["CALENDAR_PATH"])
    
    #############  Make Weights
    day_list = []
    for i in range(1942, 1970):
        day_list.append("d_{}".format(i))
    for day in day_list:
        train_val_df_origin[day] = np.nan
    
    for category in CATEGORY_ID:
        
        train_val_df = train_val_df_origin[train_val_df_origin[f"{category_name}_id"]==category]
    
    
        melt_sales = pd.melt(train_val_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name="day", value_name="sales")
        calendar_df['date'] = pd.to_datetime(calendar_df['date'], format='%Y-%m-%d')
        melt_sales = melt_sales.merge(calendar_df, left_on="day", right_on="d")
        melt_sales = pd.merge(melt_sales, sell_price_df, on=["store_id", "item_id", "wm_yr_wk"], how='left')
    
        melt_sales = melt_sales[["id", "date", "sales", "sell_price"]]
    
        melt_sales = reduce_mem_usage(melt_sales)
    
        ############## Creat mat
        # train_val_df = train_val_df[train_val_df["distribution"]=="poisson"]
    
        NUM_ITEMS = train_val_df.shape[0]
    
        product = train_val_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    
        # Weight_MAT
        weight_mat = np.c_[np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                           pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                           pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values,
                           np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
                           ].T
    
        np.save(jsn["WEIGHTS_DIR"] + f"weight_mat_{category}.npy" ,weight_mat)
        weight_mat_csr = csr_matrix(weight_mat)
    
    #     del train_val_df, sell_price_df, calendar_df, weight_mat
    #     gc.collect()
    
        ############## Loss function weights are calculated and stored.
        weight1, weight2 = weight_calc(melt_sales, product, weight_mat_csr, category)
        # del df, train_val_df; gc.collect()
    #     del melt_sales; gc.collect()
    
        ############## SAVE WEIGHT
    
        np.save(jsn["WEIGHTS_DIR"] + f"weight1_{category}.npy" ,weight1)
        np.save(jsn["WEIGHTS_DIR"] + f"weight2_{category}.npy" ,weight2)