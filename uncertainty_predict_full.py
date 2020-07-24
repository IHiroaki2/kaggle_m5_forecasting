#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:26:47 2020

@author: hiroaki_ikeshita
"""


############## import
######################################################
import numpy as np
import pandas as pd
import json
import sys

args = sys.argv

ACC_FILE_NAME = args[1]
SAVE_FILE_NAME = args[2]


file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)

submission_no = 1

qs_dict = {0.005: 84,
           0.025: 80,
           0.165: 56,
           0.250: 42,}

############## data download
######################################################
train_df = pd.read_csv(jsn["TRAIN_DATA_PATH"])
sample_submission = pd.read_csv(jsn["UNCERT_SAMPLE_SUB_PATH"])
cf = pd.read_pickle(jsn["WEIGHTS_DIR"] + "weight_size_change_format.pkl")


############## make index list
######################################################
product = train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
product["store_item_id"] = product["item_id"] + "_" + product["store_id"]

index_list = ["Total_X"]
state = list(pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').columns)
store = list(pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').columns)
cat = list(pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').columns)
dept = list(pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').columns)
state_cat = list(pd.get_dummies(product.state_id.astype(str) + "_" + product.cat_id.astype(str),drop_first=False).astype('int8').columns)
state_dept = list(pd.get_dummies(product.state_id.astype(str) + "_" + product.dept_id.astype(str),drop_first=False).astype('int8').columns)
store_cat = list(pd.get_dummies(product.store_id.astype(str) + "_" + product.cat_id.astype(str),drop_first=False).astype('int8').columns)
store_dept = list(pd.get_dummies(product.store_id.astype(str) + "_" + product.dept_id.astype(str),drop_first=False).astype('int8').columns)
item = list(pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').columns)
state_item = list(pd.get_dummies(product.state_id.astype(str) + "_" + product.item_id.astype(str),drop_first=False).astype('int8').columns)
store_item = list(product["store_item_id"].values)

for category in [state, store, cat, dept, state_cat, state_dept, store_cat, store_dept, item, state_item, store_item]:
    if category in [state, store, cat, dept, item]:
        X = []
        for word in category:
            X.append(word + "_X")
        index_list = index_list + X
        
    elif category in [state_cat, state_dept, store_cat, store_dept, state_item, store_item]:
        index_list = index_list + category
        


############## data processing
######################################################        
file_list = {"shift1":["public_pred_shift1.pkl", "validation_pred_shift1.pkl", "semival_pred_shift1.pkl"],
             "shift7":["public_pred_shift7.pkl", "validation_pred_shift7.pkl", "semival_pred_shift7.pkl"],
             "shift14":["public_pred_shift14.pkl", "validation_pred_shift14.pkl", "semival_pred_shift14.pkl"],
             "shift21":["public_pred_shift21.pkl", "validation_pred_shift21.pkl", "semival_pred_shift21.pkl"],
             "shift28":["public_pred_shift28.pkl", "validation_pred_shift28.pkl", "semival_pred_shift28.pkl"]}


for shift, file_names in file_list.items():
    pred_shift = pd.DataFrame()
    i = 0
    for file_path in file_names:
        df = pd.read_pickle(jsn["PREDICTION_DIR"] + file_path)
        df["id_serial"] = np.tile(range(30490), 28)
        df["diff"] = df["pred"] - df["target"]
        df = df.pivot_table(index=["id_serial", "id"], columns="d_serial", values="diff", aggfunc=np.sum)
        df = np.dot(cf.values, df.values)
        if i==0:
            pred_shift = df
        else:
            pred_shift = np.concatenate([pred_shift, df], axis=1)
        i+=1

    pred_shift = np.concatenate([pred_shift, np.zeros((42840, 1))], axis=1)
    pred_shift = np.sort(np.abs(pred_shift), axis=1)
    pred_shift = pd.DataFrame(pred_shift, index=index_list, columns=range(28*3+1))
    pred_shift.to_pickle(jsn["ACC_PRED_SUMMARY_DIR"] + f"{shift}.pkl")
    
    

############## make submission
######################################################   
submission = pd.read_csv(jsn["ACC_SUB_DIR"] + ACC_FILE_NAME) ###### submissionの並び順に気を付ける
submission = submission.iloc[:30490, :]
submission = submission.drop("id", axis=1)

submission = np.dot(cf.values, submission.values)
day_cols = [f"F{i}" for i in range(1, 29)]
index_list_modi = []
for word in index_list:
    index_list_modi.append(word + "_0.500_evaluation")
sample_df = pd.DataFrame(submission, index=index_list_modi, columns=day_cols)


for i in range(28):
    print(f"START_day{i+1}")
    if 0<= i <= 3:
        pred_shift = pd.read_pickle(jsn["ACC_PRED_SUMMARY_DIR"] + "shift1.pkl")
    elif 4 <= i <= 10:
        pred_shift = pd.read_pickle(jsn["ACC_PRED_SUMMARY_DIR"] + "shift7.pkl")
    elif 11 <= i <= 17:
        pred_shift = pd.read_pickle(jsn["ACC_PRED_SUMMARY_DIR"] + "shift14.pkl")
    elif 18 <= i <= 24:
        pred_shift = pd.read_pickle(jsn["ACC_PRED_SUMMARY_DIR"] + "shift21.pkl")
    elif 25 <= i <= 27:
        pred_shift = pd.read_pickle(jsn["ACC_PRED_SUMMARY_DIR"] + "shift28.pkl")
    
    for point, col in qs_dict.items():
        
        under_name = []
        for word in pred_shift.index:
            under_name.append(word + "_" + "{:.3f}".format(point) + "_evaluation")
            
        over_name = []
        for word in pred_shift.index:
            over_name.append(word + "_" + "{:.3f}".format(1 - point) + "_evaluation")
            
        under = sample_df.iloc[:, i].values - pred_shift.loc[:, col].values
        over = sample_df.iloc[:, i].values + pred_shift.loc[:, col].values
        under_df = pd.DataFrame(under, index=under_name, columns=[f"F{i+1}"])
        over_df = pd.DataFrame(over, index=over_name, columns=[f"F{i+1}"])
        qs_df = pd.concat((under_df, over_df), axis=0)
        if point == 0.005:
            day_pred = qs_df
        else:
            day_pred = pd.concat((day_pred, qs_df), axis=0)
    if i == 0:
        result_df = day_pred
    else:
        result_df = pd.concat((result_df, day_pred), axis=1)
                         
result_df = pd.concat((result_df, sample_df), axis=0)
result_df.iloc[:, :] = np.where(result_df<0, 0, result_df)


############## save submission
######################################################   
pred_1_sub = result_df.reset_index()
cols = ["id"] +  [f"F{i}" for i in range(1, 29)]
pred_1_sub.columns = cols
eval_pred = pred_1_sub.copy()
eval_pred["id"] = eval_pred["id"].apply(lambda x: x.replace("evaluation", "validation"))
submission = pd.concat((pred_1_sub, eval_pred), axis=0)
submission.to_csv(jsn["UNCERT_SUB_DIR"] + SAVE_FILE_NAME, index = False)