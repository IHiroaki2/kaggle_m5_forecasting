#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:28:14 2020

@author: hiroaki_ikeshita
"""


import pandas as pd
import json
import gc
import pickle
from datetime import datetime

file_path = "SETTINGS_FULL.json"

CAT_FEATURES = ['id_serial', 'cat_id', 'dept_id','event_name_1', 'item_id', "wday", "day", 
                'store_id','state_id', "sell_price_minority12", #"year", "month",
                "event_type_statecat_labelenc", "moon"]# "wday_day_labelenc"] #"day_y_snap_labelenc", "wday_snap_labelenc", , #"week_of_month", "week_of_year"]

CATEGORY_ID = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]  ####
category_name = "store"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)


######### prediction
#############################################


for term in ["public", "validation", "semival"]:
    for SHIFT_DAY in [1, 7, 14, 21, 28]:
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

        day_from = 1
        day_to = 28

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
        
        MODEL_FEATURES = list(df.columns)
        for elem in ["id", "date", "sales"]:
            MODEL_FEATURES.remove(elem)
        
        validation_pred = pd.read_pickle(jsn["PREDICTION_DIR"] + f"{term}_pred_shift{SHIFT_DAY}.pkl")
        
        PRED_DAY_TERM = [day_from, day_to]
        
        eval_x = df[(df['date'] >= EVAL_START_DATE) & (df["date"] <= EVAL_END_DATE)]
        
        test_df = eval_x.copy()
        
        # day_mask = (sales_df['d_serial']>=(END_TRAIN+PRED_DAY_TERM[0]))&(sales_df['d_serial']<=(END_TRAIN+PRED_DAY_TERM[1]))
        
        print("predict")
        
        for category in CATEGORY_ID:
            with open(jsn["MODEL_DIR"] + f'model_{term}_day{SHIFT_DAY}_{category}.pkl', 'rb') as fin:
                model = pickle.load(fin)
        
            day_mask = (validation_pred[f"{category_name}_id"]==category)&(validation_pred.d_serial>=(END_TRAIN+PRED_DAY_TERM[0]))&(validation_pred.d_serial<=(END_TRAIN+PRED_DAY_TERM[1]))
        
            validation_pred.loc[(validation_pred[f"{category_name}_id"]==category)&(validation_pred.d_serial>=(END_TRAIN+PRED_DAY_TERM[0]))&(validation_pred.d_serial<=(END_TRAIN+PRED_DAY_TERM[1])),"pred"] = model.predict(test_df.loc[day_mask, MODEL_FEATURES])
        
            print(category)
        
        validation_pred.to_pickle(jsn["PREDICTION_DIR"] + f"{term}_pred_shift{SHIFT_DAY}.pkl")
        

        del df, validation_pred
        gc.collect()