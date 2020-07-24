#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:53:58 2020

@author: hiroaki_ikeshita
"""


import pandas as pd
import json

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)


def pred_fm():
    
    ############### pred format
    #################################################
    
    ID = pd.read_pickle(jsn["FEATURE_BASIC_DIR"] + "id.pkl")
    store_id = pd.read_pickle(jsn["FEATURE_BASIC_DIR"] + "store_id.pkl")
    d_serial = pd.read_pickle(jsn["FEATURE_BASIC_DIR"] + "d_serial.pkl")
    shift_28_rolling_365 = pd.read_pickle(jsn["FEATURE_TARGET_DIR"] + "shift_28_roll_365.pkl")
    target= pd.read_pickle(jsn["FEATURE_TARGET_DIR"] + "sales_residual_diff_28_roll_365.pkl")
    predictions = pd.DataFrame({"id":ID,
                                 "store_id":store_id,
                                 "d_serial":d_serial,
                                 "shift_28_rolling_365":shift_28_rolling_365,
                                 "target":target})
    
    predictions["total_id"] = "total"
    predictions["pred"] = 0
    
    pred_terms = ["private", "public", "validation", "semival"]
    
    for i, term in enumerate(pred_terms):
        DAYS_COEF = i
        predictions_term = predictions.loc[(predictions.d_serial >= 1942-28*DAYS_COEF)&(predictions.d_serial <= 1969-28*DAYS_COEF)]
        predictions_term.to_pickle(jsn["PREDICTION_FM_DIR"] + f"{term}_pred_fm.pkl")
    
    
    val_terms = ["public", "validation", "semival"]
    for term in val_terms:
        for i in [1, 7, 14, 21, 28]:
            predictions_term = pd.read_pickle(jsn["PREDICTION_FM_DIR"] + f"{term}_pred_fm.pkl")
            predictions_term.to_pickle(jsn["PREDICTION_DIR"] + f"{term}_pred_shift{i}.pkl")
    
    predictions_term = pd.read_pickle(jsn["PREDICTION_FM_DIR"] + "private_pred_fm.pkl")
    predictions_term.to_pickle(jsn["PREDICTION_DIR"] + "private_pred.pkl")