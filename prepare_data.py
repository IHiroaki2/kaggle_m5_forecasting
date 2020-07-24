#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:30:19 2020

@author: hiroaki_ikeshita
"""


import make_features
import make_weights
import make_dataframe
import pred_format


CATEGORY_ID = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]  ####
category_name = "store"


### make features
print("----MAKE FEATURES-----------")
print("---BASIC---")
make_features.basic()
print("---TARGET---")
make_features.target()
print("---ADDITIONAL FEATURES---")
make_features.additional_features()
print("---ENCODING---")
make_features.encoding()
print("---LAG---")
make_features.lag()


### make weights
print("----MAKE WEIGHTS-----------")
make_weights.total_weight()
make_weights.weight_by_store()


### make dataframe
print("----MAKE DATAFRAME-----------")
make_dataframe.make_base_df()
make_dataframe.make_lag_df()


### pred format
print("----MAKE PREDICTION FORMAT-----------")
pred_format.pred_fm()
