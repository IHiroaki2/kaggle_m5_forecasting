#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:59:32 2020

@author: hiroaki_ikeshita
"""

import pandas as pd
import numpy as np
import json
import gc
import pickle
from datetime import datetime
import sys


CAT_FEATURES = ['id_serial', 'cat_id', 'dept_id','event_name_1', 'item_id', "wday", "day", 
                'store_id','state_id', "sell_price_minority12", #"year", "month",
                "event_type_statecat_labelenc", "moon"]# "wday_day_labelenc"] #"day_y_snap_labelenc", "wday_snap_labelenc", , #"week_of_month", "week_of_year"]

CATEGORY_ID = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]  ####
category_name = "store"

args = sys.argv

SAVE_FILE_NAME = args[1]

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)


######### prediction
#############################################


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
        
        MODEL_FEATURES = list(df.columns)
        for elem in ["id", "date", "sales"]:
            MODEL_FEATURES.remove(elem)
        
        private_pred = pd.read_pickle(jsn["PREDICTION_DIR"] + "private_pred.pkl")
        
        PRED_DAY_TERM = [day_from, day_to]
        
        eval_x = df[(df['date'] >= EVAL_START_DATE) & (df["date"] <= EVAL_END_DATE)]
        
        test_df = eval_x.copy()
        
        # day_mask = (sales_df['d_serial']>=(END_TRAIN+PRED_DAY_TERM[0]))&(sales_df['d_serial']<=(END_TRAIN+PRED_DAY_TERM[1]))
        
        print("predict")
        
        for category in CATEGORY_ID:
            with open(jsn["MODEL_DIR"] + f'model_{term}_day{SHIFT_DAY}_{category}.pkl', 'rb') as fin:
                model = pickle.load(fin)
        
            day_mask = (private_pred[f"{category_name}_id"]==category)&(private_pred.d_serial>=(END_TRAIN+PRED_DAY_TERM[0]))&(private_pred.d_serial<=(END_TRAIN+PRED_DAY_TERM[1]))
        
            private_pred.loc[(private_pred[f"{category_name}_id"]==category)&(private_pred.d_serial>=(END_TRAIN+PRED_DAY_TERM[0]))&(private_pred.d_serial<=(END_TRAIN+PRED_DAY_TERM[1])),"pred"] = model.predict(test_df.loc[day_mask, MODEL_FEATURES])
        
            print(category)
        
        private_pred.to_pickle(jsn["PREDICTION_DIR"] + "private_pred.pkl")


######### submit format
#############################################

private_pred = pd.read_pickle(jsn["PREDICTION_DIR"] + "private_pred.pkl")

private_pred["modi"] = private_pred["shift_28_rolling_365"] + private_pred["pred"]
private_pred["No"] = np.tile(range(30490), 28)
private_pred_pivot = private_pred.pivot_table(index=["No", "id"], columns="d_serial", values="modi", aggfunc=np.sum)
private_pred_pivot = private_pred_pivot.reset_index()
private_pred_pivot = private_pred_pivot.drop("No", axis=1)
private_pred_pivot.columns = ["id"] + [f"F{i}" for i in range(1, 29)]
eval_pred = private_pred_pivot.copy()
eval_pred["id"] = eval_pred["id"].apply(lambda x: x.replace("validation", "evaluation"))
submission = pd.concat((private_pred_pivot, eval_pred), axis=0)
submission.to_csv(jsn["ACC_SUB_DIR"] + SAVE_FILE_NAME, index = False)

