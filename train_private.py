#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:05:22 2020

@author: hiroaki_ikeshita
"""


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
from datetime import datetime
import lightgbm as lgb
from scipy.sparse import csr_matrix
import sys,time, warnings, psutil, random
from multiprocessing import Pool
import decimal
import holidays
import json

warnings.filterwarnings('ignore')

from myfunctions import reduce_mem_usage, seed_everything, weight_calc, data_division, graph_feature_importance, decay_learning_rate

from myfunctions import WRMSSE

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)


#seed_everything(seed=28)

CAT_FEATURES = ['id_serial', 'cat_id', 'dept_id','event_name_1', 'item_id', "wday", "day", 
                'store_id','state_id', "sell_price_minority12", #"year", "month",
                "event_type_statecat_labelenc", "moon"]# "wday_day_labelenc"] #"day_y_snap_labelenc", "wday_snap_labelenc", , #"week_of_month", "week_of_year"]

CATEGORY_ID = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]  ####
category_name = "store"


##################### Model Params
####################################################################

params = {
            'boosting_type': 'gbdt',
            'objective': "regression",
#             'alpha':0.5,
#             'tweedie_variance_power': 1.1,
#             "force_row_wise" : True,
#             'max_depth':11,
            'metric': 'custom',
#             'metric': '[rmse]',
            'subsample': 0.5,
            'subsample_freq': 1,
#             'learning_rate': 0.01,
            'learning_rate': 0.01,
            'num_leaves': 2**11-1,   
            'min_data_in_leaf': 2**12-1,
#             "lambda_l2" : 0.1,
            'feature_fraction': 0.8,
            'max_bin': 255,     
            'n_estimators': 1500,
            'boost_from_average': False,
            'verbose': -1,
#             "is_enable_sparse": False,
              }
def default_params():
        params["subsample"] = 0.5
        params["learning_rate"] = 0.01
#         params["learning_rate"] = 0.08
        params["num_leaves"] = 2**11-1
        params["min_data_in_leaf"] = 2**12-1
        params["feature_fraction"] = 0.8
        params["max_bin"] = 255
        params["n_estimators"] =1500
        
def category_param(category):
    default_params()
    if category=="CA_1":
#         params["n_estimators"] =1200
        pass

    elif category=="CA_2":
        params["num_leaves"] = 2**8-1
        params["min_data_in_leaf"] = 2**8-1
        
    elif category=="CA_3":
        params["learning_rate"] = 0.03
        params["num_leaves"] = 2**8-1
        params["min_data_in_leaf"] = 2**8-1
        params["n_estimators"] = 2300
        
    elif category=="CA_4":
        params["feature_fraction"] = 0.5
        
    elif category=="TX_1":
        pass
    elif category=="TX_2":
        pass
    elif category=="TX_3":
        pass
    elif category=="WI_1":
        pass
    elif category=="WI_2":
        params["num_leaves"] = 2**8-1
        params["min_data_in_leaf"] = 2**8-1
        params["feature_fraction"] = 0.5
    elif category=="WI_3":
        pass
    
    

##################### Train Models
####################################################################    

for term in ["private"]:
    for SHIFT_DAY in range(1, 29):
        print(f"-----day{SHIFT_DAY}-------------")
        
    
        if term == "private":
            TRAIN_END_DATE = "2016-05-22"
            TRAIN_START_DATE = "2012-03-28"
            VAL_END_DATE = "2016-05-22"
            VAL_START_DATE = "2016-04-25"
            EVAL_END_DATE = "2016-06-19"
            EVAL_START_DATE = "2016-05-23"
            DAYS_COEF = 0    
            
        elif term == "public":
            TRAIN_END_DATE = "2016-04-24"
            TRAIN_START_DATE = "2012-03-28"
            VAL_END_DATE = "2016-05-22"
            VAL_START_DATE = "2016-04-25"
            EVAL_END_DATE = "2016-05-22"
            EVAL_START_DATE = "2016-04-25"
            DAYS_COEF = 1
            
        elif term == "validation":
            TRAIN_END_DATE = "2016-03-27"
            TRAIN_START_DATE = "2012-03-28"
            VAL_END_DATE = "2016-04-24"
            VAL_START_DATE = "2016-03-28"
            EVAL_END_DATE = "2016-04-24"
            EVAL_START_DATE = "2016-03-28"
            DAYS_COEF = 2

        elif term == "semival":
            TRAIN_END_DATE = "2016-02-28"
            TRAIN_START_DATE = "2012-03-28"
            VAL_END_DATE = "2016-03-27"
            VAL_START_DATE = "2016-02-29"
            EVAL_END_DATE = "2016-03-27"
            EVAL_START_DATE = "2016-02-29"
            DAYS_COEF = 3
        
        END_TRAIN = 1941-28*DAYS_COEF
        
        dt1 = datetime.strptime(TRAIN_START_DATE,'%Y-%m-%d')
        dt2 = datetime.strptime(TRAIN_END_DATE,'%Y-%m-%d')
        dt = dt2 - dt1
        TRAIN_DAYS = dt.days + 1

        day_from = SHIFT_DAY
        day_to = SHIFT_DAY

        print("load data")
        base_df = pd.read_pickle(jsn["DATAFRAME_DIR"] + f"data_base_{term}_df.pkl")
        lag_df = pd.read_pickle(jsn["DATAFRAME_DIR"] + f"data_day{SHIFT_DAY}_df.pkl")
        df = pd.concat((base_df, lag_df), axis=1)

        del base_df, lag_df
        gc.collect()

        print("category feat")
        for col in CAT_FEATURES:
            try:
                df[col] = df[col].astype('category')
            except:
                pass

        train_df = pd.read_csv(jsn["TRAIN_DATA_PATH"])
        category_id_origin = train_df[f"{category_name}_id"]  ####

        del train_df,
        gc.collect()
        
        WEIGHT_SCALED_30490 = np.load(jsn["WEIGHTS_DIR"] + "WEIGHT_SCALED_30490.npy")

        for category in CATEGORY_ID:
            print(f"START_{category}")
            category_param(category)
            df_category = df[df[f"{category_name}_id"]==category] ####

            category_mask = category_id_origin == category

            weight_category = WEIGHT_SCALED_30490[category_mask]

            NUM_ITEMS = len(weight_category)

            # lgb_w_tr, lgb_w_val, lgb_w_eval = train_weight(df,)
            lgb_w_tr = np.tile(weight_category, TRAIN_DAYS)
            tr_x, tr_y, val_x, val_y, eval_x, eval_x_id_date = data_division(df_category, TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, EVAL_START_DATE, EVAL_END_DATE)

            weight1 = np.load(jsn["WEIGHTS_DIR"] + f"weight1_{category}.npy")
            weight2 = np.load(jsn["WEIGHTS_DIR"] + f"weight2_{category}.npy")
            weight_mat = np.load(jsn["WEIGHTS_DIR"] + f"weight_mat_{category}.npy")
            weight_mat_csr = csr_matrix(weight_mat)
            
            W = WRMSSE(weight1, weight2, weight_mat_csr)

            # eval_x = eval_x.drop(DROP_FEATURES_ROLL, axis=1)
            # eval_x = eval_x.drop(DROP_FEATURES_SHIFT, axis=1)

            MODEL_FEATURES = tr_x.columns

            del df_category,
            gc.collect()

            #Dataset作成
            train_set = lgb.Dataset(tr_x, tr_y, weight=lgb_w_tr) #categorical_feature = CAT_FEATURES)  # weight=lgb_w_tr, 
            val_set = lgb.Dataset(val_x, val_y,) #categorical_feature = CAT_FEATURES)

            del tr_x, tr_y, lgb_w_tr,
            gc.collect()


            ########################### Train
            #################################################################################

            SAVE_MODEL_PATH = jsn["MODEL_DIR"] + f'model_{term}_day{SHIFT_DAY}_{category}.pkl'

            # with open('/Volumes/Extreme SSD/kaggle/Walmart/pkl_data/model/model_87_stats28_roll28_4year_WRMSSE_lag_shift1_2_7.pkl', 'rb') as fin:
            #     init_model = pickle.load(fin)

            model = lgb.train(params, 
                              train_set, 
            #                   num_boost_round = 10000, 
#                               early_stopping_rounds=100, 
                              valid_sets = [train_set, val_set], 
                              verbose_eval = 100,
            #                   init_model=init_model,
        #                       fobj=obj_wrmsse4,
                              feval=W.wrmsse,
            #                   callbacks=[lgb.reset_parameter(learning_rate=decay_learning_rate)]
                             )

            with open(SAVE_MODEL_PATH, 'wb') as fout:
                pickle.dump(model, fout)


#            fi = pd.DataFrame(model.feature_importance(importance_type='gain'), index=MODEL_FEATURES, columns=["importances"])
#            graph_feature_importance(fi)


        del df,
        gc.collect()