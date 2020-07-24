#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:27:00 2020

@author: hiroaki_ikeshita
"""


import pandas as pd
import json
from  myfunctions import reduce_mem_usage

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)



def make_base_df():
    '''
    Create a basic data frame for training

    '''
    ################################# Make DataFrame
    #################################################################################
    for term in ["private", "public", "validation", "semival"]:
        
        print(f"make Basic-dataframe for {term}")
        ##Basic
        #folder select
        folder1 = jsn["FEATURE_BASIC_DIR"]
        folder2 = jsn["FEATURE_ENCODING_DIR"]
        folder4 = jsn["FEATURE_LAG_DIR"]
    
        df = pd.DataFrame([])
    
        ######file select
        ## BASIC
        flist1 = [
                 'month.pkl',
                 'day.pkl',
                 'price_momentum_y_item_store.pkl',
    #              'w_serial.pkl',
                 'item_id.pkl',
    #              'd_serial.pkl',
    #              'snap_TX.pkl',
    #              '.DS_Store',
                 'last_sales.pkl',
    #              'snap_CA.pkl',
                 'is_weekend.pkl',
    #              'sales.pkl',
                 'sell_start_log.pkl',
                 'sell_price_minority12.pkl',
                 'week_of_year.pkl',
                 'price_momentum_m_item_state.pkl',
                 'Mothers_day.pkl',
                 'year.pkl',
                 'IndependenceDay.pkl',
                 'olympic_president_elec_year.pkl',
                 'event_name_1.pkl',
                 'dept_id.pkl',
                 'event_name_2.pkl',
                 'price_unique_item_state.pkl',
                 'date.pkl',
                 'id.pkl',
                 'moon.pkl',
                 'sell_price.pkl',
                 'national_holiday.pkl',
                 'state_id.pkl',
    #              'event_type_1.pkl',
                 'price_momentum_y_item_state.pkl',
                 'event_type_statecat_labelenc.pkl',
                 'event_type_2.pkl',
                 'store_id.pkl',
                 'snap_total.pkl',
                 'cat_id.pkl',
    #              'snap_WI.pkl',
                 'price_unique_item_store.pkl',
                 'sell_start.pkl',
    #              'weekday.pkl',
                 'week_of_month.pkl',
                 'Easter.pkl',
                 'wday.pkl',
                 'price_momentum_m_item_store.pkl',
                 'OrthodoxEaster.pkl',
                 'id_serial.pkl',
                 'Ramadan_Starts.pkl',
                 'NBA_finals.pkl',
    #              'wm_yr_wk.pkl',
    #              'd.pkl',
        ]
    
    
        # Encoding
        flist2 = []
        for level in [f"LEVEL{i}" for i in range(2, 13)]:
            flist2.append(f"{term}_sales_residual_diff_28_roll_365_enc_{level}_mean.pkl")
            flist2.append(f"{term}_sales_residual_diff_28_roll_365_enc_{level}_std.pkl")
        for diff in [28]:
            for level in [f"LEVEL{i}" for i in range(2, 13)]:
                for wd in ["week", "day"]:
                    for func in ["mean", "std"]:
                        flist2.append(f"{term}_sales_residual_diff_{diff}_roll_365_enc_{wd}_{level}_{func}.pkl")
    
    
        folders = [
            folder1,
            folder2, 
    #         folder3,
        #     folder4,
        ]
    
        flists = [
            flist1,
            flist2,
    #         flist3,
        #     flist4,
        ]
    
        #make DATAFRAME
        for folder, flist in zip(folders, flists):
            for filename in flist:
                row = pd.read_pickle(folder + filename)
                filename = filename[:-4]
                df[filename] = row
            print("{}".format(folder))
    
        sales = pd.read_pickle(jsn["FEATURE_TARGET_DIR"] + "sales_residual_diff_28_roll_365.pkl")
        df["sales"] = sales
    
        df = reduce_mem_usage(df)
        
        
        order = ['id_serial', 'id', 'is_weekend', 'sell_start', 'date', 'snap_total', 'day', 'sell_price', 'event_name_1', 'week_of_month', 'wday', 'week_of_year',
                'sell_start_log', 'Mothers_day', 'national_holiday', 'NBA_finals', 'sell_price_minority12', 'year', 'month', 'olympic_president_elec_year',
                'OrthodoxEaster', 'store_id', 'moon', 'cat_id', 'Ramadan_Starts', 'IndependenceDay', 'last_sales', 'event_name_2','event_type_2', 
                'event_type_statecat_labelenc', 'Easter', 'item_id', 'dept_id', 'state_id',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL2_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL2_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL3_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL3_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL4_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL4_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL5_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL5_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL6_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL6_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL7_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL7_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL8_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL8_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL9_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL9_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL10_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL10_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL11_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL11_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL12_mean', f'{term}_sales_residual_diff_28_roll_365_enc_LEVEL12_std',
                'price_unique_item_state', 'price_momentum_m_item_state', 'price_momentum_y_item_state',
                'price_unique_item_store', 'price_momentum_m_item_store', 'price_momentum_y_item_store',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL2_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL2_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL2_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL2_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL3_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL3_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL3_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL3_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL4_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL4_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL4_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL4_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL5_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL5_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL5_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL5_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL6_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL6_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL6_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL6_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL7_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL7_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL7_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL7_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL8_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL8_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL8_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL8_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL9_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL9_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL9_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL9_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL10_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL10_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL10_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL10_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL11_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL11_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL11_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL11_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL12_mean', f'{term}_sales_residual_diff_28_roll_365_enc_week_LEVEL12_std',
                f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL12_mean', f'{term}_sales_residual_diff_28_roll_365_enc_day_LEVEL12_std',
                'sales',]
    
        df = df[order]
        
        df.to_pickle(jsn["DATAFRAME_DIR"] + f"data_base_{term}_df.pkl")

    
        print(f"Base_DataFrame_{term}")
        

def make_lag_df():
    '''
    Create data frames for Lag features
    Create 28 dataframes from day 1 to day 28
    '''      
    ################################# Make DataFrame
    #################################################################################
    for SHIFT_DAY in range(1, 29):
        
        if (SHIFT_DAY >=1) & (SHIFT_DAY<=7):
            LAG_DAY = 7
        elif (SHIFT_DAY >=8) & (SHIFT_DAY<=14):
            LAG_DAY = 14
        elif (SHIFT_DAY >=15) & (SHIFT_DAY<=21):
            LAG_DAY = 21
        elif (SHIFT_DAY >=22) & (SHIFT_DAY<=28):
            LAG_DAY = 28
        
        ##Basic
        #folder select
        folder1 = jsn["FEATURE_BASIC_DIR"]
        folder2 = jsn["FEATURE_ENCODING_DIR"]
        folder4 = jsn["FEATURE_LAG_DIR"]
    
        df = pd.DataFrame([])
    
        ######file select
        
        print(f"make Lag-dataframe for day{SHIFT_DAY}")
        if SHIFT_DAY in [7, 14, 21, 28]:
            flist4 = []
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [2, 3, 4, 8, 12]:
                    for func in ["mean",]:
                        flist4.append(f"multi_7_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_{func}.pkl")
                        
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [4, 8,]:
                    for func in ["max", "min"]:
                        flist4.append(f"multi_7_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_{func}.pkl")
                        
    
            for shift in [SHIFT_DAY,]:
                for multi in [2, 3, 5]:
                    for roll in [3, 6, 10]:
                        for func in ["mean",]:
                            flist4.append(f"multi_{multi}_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_{func}.pkl")      
                            
                            
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [7, 14, 30, 60]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_mean.pkl")
                    
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_std.pkl")
                    
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_median.pkl")
                    
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_max.pkl")
            
            for shift in [LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_min.pkl")
                                
            s = SHIFT_DAY
            for shift in [s, s+1, s+2, s+3, s+4, s+5, s+6, s+7, s+8, s+9, s+10, s+11, s+12, s+13, ]:
                flist4.append(f"sales_residual_diff_28_roll_365_shift_{shift}.pkl")
        else:
            
            flist4 = []
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [2, 3, 4, 8, 12]:
                    for func in ["mean",]:
                        flist4.append(f"multi_7_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_{func}.pkl")
                        
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [4, 8,]:
                    for func in ["max", "min"]:
                        flist4.append(f"multi_7_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_{func}.pkl")
                        
    
            for shift in [SHIFT_DAY,]:
                for multi in [2, 3, 5]:
                    for roll in [3, 6, 10]:
                        for func in ["mean",]:
                            flist4.append(f"multi_{multi}_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_{func}.pkl")      
                            
                            
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [7, 14, 30, 60]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_mean.pkl")
                    
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_std.pkl")
                    
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_median.pkl")
                    
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_max.pkl")
            
            for shift in [SHIFT_DAY, LAG_DAY, LAG_DAY+7]:
                for roll in [7, 30]:
                    flist4.append(f"multi_1_sales_residual_diff_28_roll_365_shift_{shift}_roll_{roll}_min.pkl")
                                
            s = SHIFT_DAY
            for shift in [s, s+1, s+2, s+3, s+4, s+5, s+6, s+7, s+8, s+9, s+10, s+11, s+12, s+13, ]:
                flist4.append(f"sales_residual_diff_28_roll_365_shift_{shift}.pkl")
    
    
    
        folders = [
    #         folder1,
    #         folder2, 
    #         folder3,
            folder4,  
        ]
    
        flists = [
    #         flist1,
    #         flist2,
    #         flist3,
            flist4,
        ]
    
    
        #make DATAFRAME
        for folder, flist in zip(folders, flists):
            for filename in flist:
                row = pd.read_pickle(folder + filename)
                filename = filename[:-4]
                df[filename] = row
            print("{}".format(folder))
    
    #     sales = pd.read_pickle("features/Target/sales_residual_diff_28_roll_365.pkl")
    #     df["sales"] = sales
    
        df = reduce_mem_usage(df)
    
    #     df.to_pickle(f"dataframe/data_base_{term}_df.pkl")
        df.to_pickle(jsn["DATAFRAME_DIR"] + f"data_day{SHIFT_DAY}_df.pkl")
    
        print(f"Lag_DataFrame_day{SHIFT_DAY}")