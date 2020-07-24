#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:30:17 2020

@author: hiroaki_ikeshita
"""


##################### PREPARE
####################################################################

##################### Import
####################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
import gc
import os
import pdb
import math
import pickle
from sympy import *
from datetime import datetime
from scipy.sparse import csr_matrix
import sys,time, warnings, psutil, random
from multiprocessing import Pool
import decimal
import holidays
import json

warnings.filterwarnings('ignore')



CATEGORY_ID = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]  ####
category_name = "store"
NUM_ITEMS = 3049

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)

##################### some function
####################################################################

## select seed
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

##################### save memory
####################################################################

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: 
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


##################### Make data
####################################################################

def make_df(path, features, VAL_START_DATE):
    for i, filename in enumerate(features):
        row = pd.read_pickle(path + filename + ".pkl")
        if i == 0:
            df = pd.DataFrame({filename:row})
        else:
            df[filename] = row
            
            
##################### data splitting
####################################################################

def data_division(df,TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, EVAL_START_DATE, EVAL_END_DATE):
    
    train = df[(df['date'] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)]
    val = df[(df['date'] >= VAL_START_DATE) & (df["date"] <= VAL_END_DATE)]
    evaluation = df[(df['date'] >= EVAL_START_DATE) & (df["date"] <= EVAL_END_DATE)]
    evaluation_id_date = evaluation
    train = train.drop(["id", "date"], axis=1)
    val = val.drop(["id", "date"], axis=1)
    evaluation = evaluation.drop(["id", "date"], axis=1)
    
    tr_y = train["sales"]
    tr_x = train.drop("sales", axis=1)
    val_y = val["sales"]
    val_x = val.drop("sales", axis=1)
    eval_x = evaluation_id_date
    eval_x_id_date = evaluation_id_date.drop("sales", axis=1)

    return tr_x, tr_y, val_x, val_y, eval_x, eval_x_id_date

##################### feature importance graph
####################################################################

def graph_feature_importance(df):
    df = df.sort_values("importances", ascending=True)
    plt.figure(figsize=(16, 50))
    plt.barh([i for i in range(len(df.index))], df["importances"])
    plt.yticks([i for i in range(len(df.index))], df.index)
    plt.title("feature_importance")
    plt.show()
    
    
##################### decay_learning rate
####################################################################

def decay_learning_rate(current_iter):
    if current_iter < 200:
        lr = 0.03
    elif current_iter < 300:
        lr = 0.01
#     elif current_iter < 500:
#         lr = 0.01
#     elif current_iter < 1000:
#         lr = 0.008
#     elif current_iter < 1500:
#         lr = 0.03
#     elif current_iter < 3000:
#         lr = 0.02
    else:
        lr = 0.005
    return lr

########################### Helper to make dynamic rolling lags
#################################################################################

def make_lag(LAG_DAY):
    base_eval = eval_x[['id','sales']]
    col_name = 'sales_residual_diff_28_roll_365_shift_{:02d}'.format(LAG_DAY)
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return base_eval[[col_name]]


def make_lag_roll_mean(LAG_DAY,):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    base_eval = eval_x[["id", "sales"]]
    col_name = f"sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean"
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return base_eval[[col_name]]

def make_lag_roll_std(LAG_DAY,):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    base_eval = eval_x[["id", "sales"]]
    col_name = f"sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_std"
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(shift_day).rolling(roll_wind).std())
    return base_eval[[col_name]]

def make_lag_roll_max(LAG_DAY,):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    base_eval = eval_x[["id", "sales"]]
    col_name = f"sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_max"
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(shift_day).rolling(roll_wind).max())
    return base_eval[[col_name]]

def make_lag_roll_min(LAG_DAY,):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    base_eval = eval_x[["id", "sales"]]
    col_name = f"sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_min"
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(shift_day).rolling(roll_wind).min())
    return base_eval[[col_name]]

def make_lag_roll_skew(LAG_DAY,):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    base_eval = eval_x[["id", "sales"]]
    col_name = f"sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_skew"
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(shift_day).rolling(roll_wind).skew())
    return base_eval[[col_name]]

def make_lag_roll_kurt(LAG_DAY,):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    base_eval = eval_x[["id", "sales"]]
    col_name = f"sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_kurt"
    base_eval[col_name] = base_eval.groupby(['id'])["sales"].transform(lambda x: x.shift(shift_day).rolling(roll_wind).kurt())
    return base_eval[[col_name]]

##################### WRMSSE
####################################################################

##################### CUSTOM METRIC
####################################################################

def weight_calc(data,
                product,
                weight_mat_csr,
                category=None,
#                 sales_train_val
               ):

    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    
    sales_train_val = pd.read_csv(jsn["TRAIN_DATA_PATH"])
    
    if category is not None:
        sales_train_val = sales_train_val[sales_train_val[f"{category_name}_id"]==category]
    
    d_name = ['d_' + str(i+1) for i in range(1941)]
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values
    # calculate the start position(first non-zero demand observed date) for each item 
    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1942),(weight_mat_csr.shape[0],1)))
    

    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)
    
    flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1942),(weight_mat_csr.shape[0],1)))<1

    sales_train_val = np.where(flag,np.nan,sales_train_val)

    # denominator of RMSSE / RMSSEの分母
    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1942-start_no)
    
    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-04-25') & (data['date'] <= '2016-05-22')]
    df_tmp['amount'] = df_tmp['sales'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp[product.id].values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)

    del sales_train_val
    gc.collect()
    
    return weight1, weight2


class WRMSSE:
    def __init__(self, weight1, weight2, weight_mat_csr):
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight_mat_csr = weight_mat_csr
    
    def wrmsse(self, preds, data):
    #     DAYS_PRED = preds // 
        # this function is calculate for last 28 days to consider the non-zero demand period
        DAYS_PRED = 28
        # actual obserbed values 
        y_true = data.get_label()
        
        y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
        preds = preds[-(NUM_ITEMS * DAYS_PRED):]
        
        y_true = y_true.astype(np.float32)
        preds = preds.astype(np.float32)
        
        # number of columns
        num_col = DAYS_PRED
    
        # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) 
    #     pdb.set_trace()
        reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
        reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
        
              
        train = self.weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
        
        score = np.sum(
                    np.sqrt(
                        np.mean(
                            np.square(
                                train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / self.weight1) * self.weight2)
        
        return 'wrmsse', score, False

def wrmsse_simple(preds, data):
    
    # actual obserbed values 
    y_true = data.get_label()
    
    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) 
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
          
    train = np.c_[reshaped_preds, reshaped_true]
    
    weight2_2 = weight2[:NUM_ITEMS]
    weight2_2 = weight2_2/np.sum(weight2_2)
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) /  weight1[:NUM_ITEMS])*weight2_2)
    
    return 'wrmsse', score, False


##################### CUSTOM OBJECTIVE
####################################################################

def obj_wrmsse(preds, data):
    
    dt1 = datetime.strptime(TRAIN_START_DATE,'%Y-%m-%d')
    dt2 = datetime.strptime(TRAIN_END_DATE,'%Y-%m-%d')
    dt = dt2 - dt1
    TRAIN_DAYS = dt.days + 1
    
    y_true = data.get_label()

    y_true = y_true[-(NUM_ITEMS * TRAIN_DAYS):]
    preds = preds[-(NUM_ITEMS * TRAIN_DAYS):]

    reshaped_preds = preds.reshape(TRAIN_DAYS, NUM_ITEMS).T
    reshaped_true = y_true.reshape(TRAIN_DAYS, NUM_ITEMS).T
    
    #ロス計算値
    pred_fm = df_fm
#     pred_fm.iloc[:, 13:] += (reshaped_preds - reshaped_true) / np.array(weight_df1["w_store_id_&_item_id"]).reshape(-1, 1)
#     * np.array(weight_df2["w_store_id_&_item_id"]).reshape(-1, 1)
#     
    
    reshaped_preds_df = df_fm
    reshaped_true_df = df_fm
    reshaped_preds_df.iloc[:, 13:] = reshaped_preds
    reshaped_true_df.iloc[:, 13:] = reshaped_true
    
#     ps = df_fm
#     ts = df_fm
    
    for level in ["LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5", "LEVEL10"]:
#         print("Start_{}".format(level))
        days = ["d_{}".format(i+1) for i in range(TRAIN_DAYS)]
        ps = np.array(reshaped_preds_df.groupby(LEVEL[level])[days].transform(np.sum))
        ts = np.array(reshaped_true_df.groupby(LEVEL[level])[days].transform(np.sum))
#         pdb.set_trace()
        pred_fm.iloc[:, 13:] += (ps - ts) / np.array(weight_df1["w_{}".format(LEVEL[level][0])]).reshape(-1, 1) * np.array(weight_df2["w_{}".format(LEVEL[level][0])]).reshape(-1, 1)
#     
#         pdb.set_trace()
#         ps = ps.reset_index()
#         idxes = list(ps.index)
#         pdb.set_trace()
#         col = LEVEL[level][0]
        
        
#         for idx in idxes:
#             pred_fm.loc[pred_fm[col] == idx, days] += ps.loc[idx, days] - ts.loc[idx, days]
#         print("Finish_{}".format(level))
    
    y_p = np.array(pred_fm.iloc[:, 13:]).T.flatten()     

    grad = -y_p/(TRAIN_DAYS*NUM_ITEMS)
    hess = np.array([1/(TRAIN_DAYS*NUM_ITEMS) for i in range(TRAIN_DAYS*NUM_ITEMS)])

    return grad, hess

def obj_wrmsse2(preds, data):
    y = data.get_label()
    w = data.get_weight()
    yhat = preds
    grad = w*(yhat - y)
    hess = np.ones_like(yhat)
    return grad, hess

def obj_wrmsse4(preds, data):
    y = data.get_label()
#     w = data.get_weight()
    yhat = preds
    diff = yhat - y
    w = np.tile(WEIGHT_SCALED_30490, TRAIN_DAYS)
    
    diff = diff * w
    
    grad = diff
    hess = w
    
    return grad, hess

