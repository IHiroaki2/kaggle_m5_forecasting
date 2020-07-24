#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:30:18 2020

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
import sys,time, warnings, psutil, random
from multiprocessing import Pool
import decimal
import holidays
import json

from  myfunctions import reduce_mem_usage

warnings.filterwarnings('ignore')

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)

##################Basic Features
#####################################################

def basic():
    '''
    Create basic features such as ID, calendar, sell_prices

    '''
    
    train_val_df = pd.read_csv(jsn["TRAIN_DATA_PATH"])
    sell_price_df = pd.read_csv(jsn["SELL_PRICES_PATH"])
    calendar_df = pd.read_csv(jsn["CALENDAR_PATH"])
    
    ### Basic Category ID
    day_list = []
    for i in range(1942, 1970):
        day_list.append("d_{}".format(i))
    for day in day_list:
        train_val_df[day] = np.nan
        
    train_val_df["id_serial"] = list(range(30490))
        
    melt_sales = pd.melt(train_val_df, id_vars=['id_serial', 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name="d", value_name="sales")
    
    
    ####### Calendar feature
    #event2
    calendar_df.loc[calendar_df["event_name_2"]== "Cinco De Mayo", "event_name_2"] = 0
    calendar_df.loc[calendar_df["event_name_2"]== "Easter", "event_name_2"] = 1
    calendar_df.loc[calendar_df["event_name_2"]== "Father's day", "event_name_2"] = 2
    calendar_df.loc[calendar_df["event_name_2"]== "OrthodoxEaster", "event_name_2"] = 3
    calendar_df.loc[calendar_df["event_type_2"]== "Cultural", "event_type_2"] = 0
    calendar_df.loc[calendar_df["event_type_2"]== "Religious", "event_type_2"] = 1
    calendar_df["event_name_2"] = calendar_df["event_name_2"].astype(np.float16)
    calendar_df["event_type_2"] = calendar_df["event_type_2"].astype(np.float16)
    
    #weekend
    calendar_df["is_weekend"] = 0
    calendar_df.loc[calendar_df["weekday"] == "Saturday", "is_weekend"] = 1
    calendar_df.loc[calendar_df["weekday"] == "Sunday", "is_weekend"] = 1
    
    # d_serial
    calendar_df["d_serial"] = calendar_df["d"].apply(lambda x: int(x[2:]))
    
    # w_serial
    calendar_df["w_serial"] = 0
    cnt = 1
    for i in range(calendar_df.shape[0]):
        if i % 7 == 0:
            cnt += 1
        calendar_df.loc[i, "w_serial"] = cnt
    
    # date
    calendar_df['date'] = pd.to_datetime(calendar_df['date'], format='%Y-%m-%d')
    
    # day
    calendar_df["day"] = calendar_df["date"].apply(lambda x: x.day)
    calendar_df["month"] = calendar_df["date"].apply(lambda x: x.month)
    calendar_df["year"] = calendar_df["date"].apply(lambda x: x.year)
    
    # Mooon
    dec = decimal.Decimal
    
    def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
        diff = d - datetime(2001, 1, 1)
        days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
        lunations = dec("0.20439731") + (days * dec("0.03386319269"))
        phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
        return int(phase_index) % 8
    
    calendar_df['moon'] = calendar_df.date.apply(get_moon_phase)
    
    # week of month, year
    calendar_df["week_of_month"] = calendar_df["w_serial"] - calendar_df.groupby(["year", "month"])["w_serial"].transform("min") +1
    calendar_df["week_of_year"] = calendar_df["w_serial"] - calendar_df.groupby(["year"])["w_serial"].transform("min") +1
    
    
    # olympic_president_elec_year
    calendar_df["olympic_president_elec_year"] = 0
    calendar_df.loc[calendar_df["year"] == 2012, "olympic_president_elec_year"] = 1
    calendar_df.loc[calendar_df["year"] == 2016, "olympic_president_elec_year"] = 1
    
    
    # NBA_Finals
    calendar_df["NBA_finals"] = 0
    for day in ["d_123", "d_124", "d_125", "d_126", "d_127", "d_128", "d_129", "d_130", "d_131", "d_132", "d_133", "d_134", "d_135", 
              "d_501", "d_502", "d_503", "d_504", "d_505", "d_506", "d_507", "d_508", "d_509", "d_510", 
              "d_860", "d_861", "d_862", "d_863", "d_864", "d_865", "d_866", "d_867", "d_868", "d_869", "d_870", "d_871", "d_872", "d_873", "d_874", 
              "d_1224", "d_1225", "d_1226", "d_1227", "d_1228", "d_1229", "d_1230", "d_1231", "d_1232", "d_1233", "d_1234", 
              "d_1588", "d_1589", "d_1590", "d_1591", "d_1592", "d_1593", "d_1594", "d_1595", "d_1596", "d_1597", "d_1598", "d_1599", "d_1600", 
              "d_1952", "d_1953", "d_1954", "d_1955", "d_1956", "d_1957", "d_1958", "d_1959", "d_1960", "d_1961", "d_1962", "d_1963", "d_1964", 
              "d_1965", "d_1966", "d_1967", "d_1968", "d_1969"]:
        calendar_df.loc[calendar_df["d"] == day  , "NBA_finals"] = 1
    
        
    # Ramadan start
    day_list = []
    for i in range(30):
        day_list.append("d_{}".format(i+185))
        day_list.append("d_{}".format(i+539))
        day_list.append("d_{}".format(i+893))
        day_list.append("d_{}".format(i+1248))
        day_list.append("d_{}".format(i+1602))
    for i in range(13):
        day_list.append("d_{}".format(i+1957))
    day_list
    
    calendar_df["Ramadan_Starts"] = 0
    for day in day_list:
        calendar_df.loc[calendar_df["d"] == day, "Ramadan_Starts"] = 1
        
    # Mothers day 
    day_list_1 = ["d_100", "d_471", "d_835", "d_1199", "d_1563", "d_1927"]
    day_list_2 = ["d_99", "d_470", "d_834", "d_1198", "d_1562", "d_1926"]
    calendar_df["Mothers_day"] = 0
    for day in day_list_1:
        calendar_df.loc[calendar_df["d"] == day, "Mothers_day"] = 1
    for day in day_list_2:
        calendar_df.loc[calendar_df["d"] == day, "Mothers_day"] = 2
    
    # OrthodoxEaster
    day_list_1 = ["d_86", "d_443", "d_828", "d_1178", "d_1535", "d_1920"]
    day_list_2 = ["d_85", "d_442", "d_827", "d_1177", "d_1534", "d_1919"]
    calendar_df["OrthodoxEaster"] = 0
    for day in day_list_1:
        calendar_df.loc[calendar_df["d"] == day, "OrthodoxEaster"] = 1
    for day in day_list_2:
        calendar_df.loc[calendar_df["d"] == day, "OrthodoxEaster"] = 2
    
    # Easter
    day_list_1 = ["d_86", "d_436", "d_793", "d_1178", "d_1528", "d_1885"]
    day_list_2 = ["d_85", "d_435", "d_792", "d_1177", "d_1527", "d_1884"]
    calendar_df["Easter"] = 0
    for day in day_list_1:
        calendar_df.loc[calendar_df["d"] == day, "Easter"] = 1
    for day in day_list_2:
        calendar_df.loc[calendar_df["d"] == day, "Easter"] = 2
        
    
    # IndependenceDay
    day_list_1 = ["d_151", "d_517", "d_882", "d_1247", "d_1612"]
    day_list_2 = ["d_152", "d_518", "d_883", "d_1248", "d_1613"]
    day_list_3 = ["d_153", "d_519", "d_884", "d_1249", "d_1614"]
    day_list_4 = ["d_154", "d_520", "d_885", "d_1250", "d_1615"]
    day_list_5 = ["d_155", "d_521", "d_886", "d_1251", "d_1616"]
    day_list_6 = ["d_156", "d_522", "d_887", "d_1252", "d_1617"]
    day_list_7 = ["d_157", "d_523", "d_888", "d_1253", "d_1618"]
    day_list_8 = ["d_158", "d_524", "d_889", "d_1254", "d_1619"]
    day_list_9 = ["d_159", "d_525", "d_890", "d_1255", "d_1620"]
    day_list_10 = ["d_160", "d_526", "d_891", "d_1256", "d_1621"]
    
    calendar_df["IndependenceDay"] = 0
    for day in day_list_1:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 1
    for day in day_list_2:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 2
    for day in day_list_3:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 3
    for day in day_list_4:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 4
    for day in day_list_5:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 5
    for day in day_list_6:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 6
    for day in day_list_7:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 7
    for day in day_list_8:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 8
    for day in day_list_9:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 9
    for day in day_list_10:
        calendar_df.loc[calendar_df["d"] == day, "IndependenceDay"] = 10
    
    
    
    ###### price feature
    ##############################
    sell_price_df["state_id"] = sell_price_df["store_id"].apply(lambda x: x[:2])
    sell_price_df["dept_id"] = sell_price_df["item_id"].apply(lambda x: x[:-4])
    sell_price_df["cat_id"] = sell_price_df["dept_id"].apply(lambda x: x[:-2])
    
    sell_price_df["price_unique_item_state"] = sell_price_df.groupby(['state_id','item_id'])['sell_price'].transform('nunique')
    sell_price_df["price_unique_item_store"] = sell_price_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
    
    calendar_prices = calendar_df[['wm_yr_wk','month','year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    sell_price_df = sell_price_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
    
    sell_price_df['price_momentum_m_item_state'] = sell_price_df['sell_price'] / sell_price_df.groupby(['state_id','item_id','month'])['sell_price'].transform('mean')
    sell_price_df['price_momentum_y_item_state'] = sell_price_df['sell_price'] / sell_price_df.groupby(['state_id','item_id','year'])['sell_price'].transform('mean')
    sell_price_df['price_momentum_m_item_store'] = sell_price_df['sell_price'] / sell_price_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    sell_price_df['price_momentum_y_item_store'] = sell_price_df['sell_price'] / sell_price_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')
    
    
    
    # sell_start
    sell_price_df["one"] = 1
    sell_price_df["sell_start"] = sell_price_df.groupby(["store_id", "item_id"])["one"].transform(lambda x: x.cumsum())
    sell_price_df["sell_start_log"] = sell_price_df.groupby(["store_id", "item_id"])["sell_start"].transform(lambda x: np.log(x))
    
    sell_price_df = sell_price_df.drop(["month", "year", "one", "state_id", "dept_id", "cat_id"], axis=1)
    
    
    ####### merge
    melt_sales = melt_sales.merge(calendar_df, on="d", how="left")
    melt_sales = pd.merge(melt_sales, sell_price_df, on=["store_id", "item_id", "wm_yr_wk"], how='left')
    
    ####### save by column
    melt_sales = reduce_mem_usage(melt_sales)
    for col in melt_sales.columns:
        melt_sales[col].to_pickle(jsn['FEATURE_BASIC_DIR'] + f"{col}.pkl")
        
    del melt_sales, sell_price_df, calendar_df, train_val_df
    gc.collect()


################## Target Feature
#####################################################
  
def target():      
    ID = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "id.pkl")
    sales = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "sales.pkl")
    df = pd.DataFrame({"id": ID,
                       "sales": sales})
    
    df = reduce_mem_usage(df)
    
    df["shift_28_roll_365"] = df.groupby(["id"])["sales"].transform(lambda x: x.shift(28).rolling(365).mean())
    df["sales_residual_diff_28_roll_365"] = df["sales"] - df["shift_28_roll_365"]
    
    df["shift_28_roll_365"].to_pickle(jsn['FEATURE_TARGET_DIR'] + "shift_28_roll_365.pkl")
    df["sales_residual_diff_28_roll_365"].to_pickle(jsn['FEATURE_TARGET_DIR'] + "sales_residual_diff_28_roll_365.pkl")
    
    del df, ID, sales
    gc.collect()
    

################## Additional Features
#####################################################

def additional_features():
    
    ####### snap total
    ###############################
    state_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "state_id.pkl")
    snap_CA = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "snap_CA.pkl")
    snap_TX = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "snap_TX.pkl")
    snap_WI = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "snap_WI.pkl")
    df = pd.DataFrame({"state_id":state_id,
                       "snap_CA":snap_CA,
                       "snap_TX":snap_TX,
                       "snap_WI":snap_WI})
    
    df["snap_total"] = 0
    df.loc[df.state_id == "CA", "snap_total"] = df.loc[df.state_id == "CA", "snap_CA"]
    df.loc[df.state_id == "TX", "snap_total"] = df.loc[df.state_id == "TX", "snap_TX"]
    df.loc[df.state_id == "WI", "snap_total"] = df.loc[df.state_id == "WI", "snap_WI"]
    
    df["snap_total"].to_pickle(jsn['FEATURE_BASIC_DIR'] + "snap_total.pkl") 
    
    del df, state_id, snap_CA, snap_TX, snap_WI
    gc.collect()
    
    
    ####### event_type_statecat_labelenc
    ###############################
    train_val_df = pd.read_csv(jsn["TRAIN_DATA_PATH"])
    calendar_df = pd.read_csv(jsn["CALENDAR_PATH"])
    
    day_list = []
    for i in range(1914, 1970):
        day_list.append("d_{}".format(i))
    for day in day_list:
        train_val_df[day] = np.nan
        
    melt_sales = pd.melt(train_val_df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name="day", value_name="demand")
    melt_sales = melt_sales.merge(calendar_df, left_on="day", right_on="d")
    melt_sales = melt_sales[["state_id", "cat_id", "event_type_1"]]
    melt_sales["event_type_statecat"] = np.nan
    
    df_evtyp = melt_sales.loc[(melt_sales.event_type_1 == 'Cultural') | \
                   (melt_sales.event_type_1 == 'National') | \
                   (melt_sales.event_type_1 == 'Religious') | \
                   (melt_sales.event_type_1 == 'Sporting'), :]
    
    df_evtyp["event_type_statecat"] = df_evtyp["state_id"] + df_evtyp["cat_id"] + df_evtyp["event_type_1"]
    
    melt_sales.loc[(melt_sales.event_type_1 == 'Cultural') | \
                   (melt_sales.event_type_1 == 'National') | \
                   (melt_sales.event_type_1 == 'Religious') | \
                   (melt_sales.event_type_1 == 'Sporting'), "event_type_statecat"] = df_evtyp["event_type_statecat"].values
    
    melt_sales["event_type_statecat_labelenc"] = 0
    
    type_list = ['CAHOBBIESSporting', 'CAHOUSEHOLDSporting', 'CAFOODSSporting',
           'TXHOBBIESSporting', 'TXHOUSEHOLDSporting', 'TXFOODSSporting',
           'WIHOBBIESSporting', 'WIHOUSEHOLDSporting', 'WIFOODSSporting',
           'CAHOBBIESCultural', 'CAHOUSEHOLDCultural', 'CAFOODSCultural',
           'TXHOBBIESCultural', 'TXHOUSEHOLDCultural', 'TXFOODSCultural',
           'WIHOBBIESCultural', 'WIHOUSEHOLDCultural', 'WIFOODSCultural',
           'CAHOBBIESNational', 'CAHOUSEHOLDNational', 'CAFOODSNational',
           'TXHOBBIESNational', 'TXHOUSEHOLDNational', 'TXFOODSNational',
           'WIHOBBIESNational', 'WIHOUSEHOLDNational', 'WIFOODSNational',
           'CAHOBBIESReligious', 'CAHOUSEHOLDReligious', 'CAFOODSReligious',
           'TXHOBBIESReligious', 'TXHOUSEHOLDReligious', 'TXFOODSReligious',
           'WIHOBBIESReligious', 'WIHOUSEHOLDReligious', 'WIFOODSReligious']
    
    for num, ty in zip(range(1, 37), type_list):
        melt_sales.loc[melt_sales["event_type_statecat"] == ty, "event_type_statecat_labelenc"] = num
        
    melt_sales["event_type_statecat_labelenc"].to_pickle(jsn['FEATURE_BASIC_DIR'] + "event_type_statecat_labelenc.pkl")
    
    del melt_sales, train_val_df, calendar_df
    gc.collect()
    
    
    ###### national holiday
    #############################
    date = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "date.pkl")
    state_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "state_id.pkl")
    df = pd.DataFrame({"date":date,
                       "state_id":state_id})
    
    df["National_Holiday"] = 0
    for state in ["CA", "TX", "WI"]:
        for year in [2011, 2012, 2013, 2014, 2015, 2016]:
            for d, name in sorted(holidays.US(state=state, years=year).items()):
                df.loc[(df.state_id==state)&(df.date==d.strftime('%Y-%m-%d')), "National_Holiday"] = 1
    
    df["National_Holiday"].to_pickle(jsn['FEATURE_BASIC_DIR'] + "national_holiday.pkl")
    
    del df, date, state_id
    gc.collect()
        
        
    ###### Last Sales
    ##############################
    df = pd.read_csv(jsn["TRAIN_DATA_PATH"])
    df["last_sales"] = 0
    
    day = 1913
    for i in range(1913):
        df.loc[(df["d_{}".format(day)]>0) & (df["last_sales"] == 0), "last_sales"] = day
        if np.sum(df["last_sales"] == 0) == 0:
            break
        day -= 1
    
    day_list = []
    for i in range(1914, 1970):
        day_list.append("d_{}".format(i))
    for day in day_list:
        df[day] = np.nan
        
    melt_sales = pd.melt(df, id_vars=['id', 'item_id',
    #                                             'distribution', 
                                                'dept_id', 'cat_id', 'store_id', 'state_id', "last_sales",], var_name="day", value_name="sales")
    
    melt_sales["last_sales"].to_pickle(jsn['FEATURE_BASIC_DIR'] + "last_sales.pkl")
    
    del melt_sales, df
    gc.collect()
    
    ### minority

    sell_price = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "sell_price.pkl")
    
    df = pd.DataFrame({"sell_price":sell_price})
    
    def minority(num):
        f, i = math.modf(num)
        f = str(f)
        return f[2:4]
    
    df["sell_price_minority12"] = df["sell_price"].transform(lambda x: minority(x))
    df.loc[df["sell_price_minority12"] == "n", "sell_price_minority12"] = 9999
    df["sell_price_minority12"] = df["sell_price_minority12"].apply(lambda x: int(x))
    
    df["sell_price_minority12"].to_pickle(jsn['FEATURE_BASIC_DIR'] + "sell_price_minority12.pkl")
    
    del df, sell_price
    gc.collect()
    
    
################## Encoding Features(mean, std)
#####################################################
    
def encoding():
    
    ID = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "id.pkl")
    item_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "item_id.pkl")
    store_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "store_id.pkl")
    dept_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "dept_id.pkl")
    state_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "state_id.pkl")
    cat_id = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "cat_id.pkl")
    d_serial = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "d_serial.pkl")
    sell_price = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "sell_price.pkl")
    wday = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "wday.pkl")
    day = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "day.pkl")
    sales = pd.read_pickle(jsn['FEATURE_TARGET_DIR'] + "sales_residual_diff_28_roll_365.pkl")
    
    df = pd.DataFrame({"id":ID,
                       "item_id":item_id,
                       "store_id":store_id,
                       "dept_id":dept_id,
                       "state_id":state_id,
                       "cat_id":cat_id,
                       "d_serial":d_serial,
                       "sell_price":sell_price,
                       "wday":wday,
                       "day":day,
                       "sales":sales})
    
    df = reduce_mem_usage(df)
    
    df["flag"] = 0
    df.loc[df["sell_price"]>=0, "flag"] = 1
    df.loc[df["flag"]==0, "sales"] = np.nan
    
    pred_terms = ["private", "public", "validation", "semival"]
    
    for i, term in enumerate(pred_terms):
        
        df.loc[df["d_serial"]>=1942-i*28, "sales"] = np.nan
    
        LEVEL = {
                     "LEVEL2": ["state_id"],
                     "LEVEL3": ["store_id"],
                     "LEVEL4": ["cat_id"],
                     "LEVEL5": ["dept_id"],
                     "LEVEL6": ["state_id", "cat_id"],
                     "LEVEL7": ["state_id", "dept_id"],
                     "LEVEL8": ["store_id", "cat_id"],
                     "LEVEL9": ["store_id", "dept_id"],
                     "LEVEL10": ["item_id"],
                     "LEVEL11": ["state_id", "item_id"],
                     "LEVEL12": ["store_id", "item_id"]}
    
        for key, value in LEVEL.items():
            df.groupby(value + ["wday"])["sales"].transform(np.mean).to_pickle(jsn["FEATURE_ENCODING_DIR"] + f"{term}_sales_residual_diff_28_roll_365_enc_week_{key}_mean.pkl")
            df.groupby(value + ["wday"])["sales"].transform(np.std).to_pickle(jsn["FEATURE_ENCODING_DIR"] + f"{term}_sales_residual_diff_28_roll_365_enc_week_{key}_std.pkl")
            df.groupby(value + ["day"])["sales"].transform(np.mean).to_pickle(jsn["FEATURE_ENCODING_DIR"] + f"{term}_sales_residual_diff_28_roll_365_enc_day_{key}_mean.pkl")
            df.groupby(value + ["day"])["sales"].transform(np.std).to_pickle(jsn["FEATURE_ENCODING_DIR"] + f"{term}_sales_residual_diff_28_roll_365_enc_day_{key}_std.pkl")  
            df.groupby(value)["sales"].transform(np.mean).to_pickle(jsn["FEATURE_ENCODING_DIR"] + f"{term}_sales_residual_diff_28_roll_365_enc_{key}_mean.pkl")
            df.groupby(value)["sales"].transform(np.std).to_pickle(jsn["FEATURE_ENCODING_DIR"] + f"{term}_sales_residual_diff_28_roll_365_enc_{key}_std.pkl")
            
    del df, ID, item_id, store_id, dept_id, state_id, cat_id, d_serial, sell_price, wday, day, sales
    gc.collect()

################## Lag Features
#####################################################

def lag():
    ###### LAG Rolling
    #############################
    
    ID = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "id.pkl")
    d_serial = pd.read_pickle(jsn['FEATURE_BASIC_DIR'] + "d_serial.pkl")
    sales = pd.read_pickle(jsn['FEATURE_TARGET_DIR'] + "sales_residual_diff_28_roll_365.pkl")
    df = pd.DataFrame({"id":ID,
                       "d_serial":d_serial,
                       "sales":sales})
    df = reduce_mem_usage(df)
    
    #### shift 1~ 56
    for shift_day in range(1, 57):
        df.groupby(["id"])["sales"].transform(lambda x: x.shift(shift_day)).to_pickle(jsn["FEATURE_LAG_DIR"] + f"sales_residual_diff_28_roll_365_shift_{shift_day}.pkl")
    
    #### rolling
    ####################
    for i in [1, 2, 3, 5, 7]:
        df[f"multi_{i}"] = df["d_serial"].transform(lambda x: x%i)
    
    for shift_day in range(1, 36):
        target = f"sales_residual_diff_28_roll_365_shift_{shift_day}"
        
        df[target] = pd.read_pickle(jsn["FEATURE_LAG_DIR"] + f"sales_residual_diff_28_roll_365_shift_{shift_day}.pkl")
        #### multi 2, 3, 5 rolling
        for roll_wind in [3, 6, 10]:
            df.groupby(['id', 'multi_2'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_2_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
            df.groupby(['id', 'multi_3'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_3_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
            df.groupby(['id', 'multi_5'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_5_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
            
    
        #### multi 7 rolling
        for roll_wind in [2, 3, 4, 8, 12]:
            if roll_wind in [4, 8]:
                df.groupby(['id', 'multi_7'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_7_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
                df.groupby(['id', 'multi_7'])[target].transform(lambda x: x.rolling(roll_wind).max()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_7_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_max.pkl")
                df.groupby(['id', 'multi_7'])[target].transform(lambda x: x.rolling(roll_wind).min()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_7_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_min.pkl")
            else:
                df.groupby(['id', 'multi_7'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_7_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
    
        #### multi 1 rolling
        for roll_wind in [7, 14, 30, 60]: 
            if roll_wind in [7, 30]:
                df.groupby(['id', 'multi_1'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_1_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
                df.groupby(['id', 'multi_1'])[target].transform(lambda x: x.rolling(roll_wind).std()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_1_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_std.pkl")
                df.groupby(['id', 'multi_1'])[target].transform(lambda x: x.rolling(roll_wind).max()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_1_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_max.pkl")
                df.groupby(['id', 'multi_1'])[target].transform(lambda x: x.rolling(roll_wind).min()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_1_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_min.pkl")
                df.groupby(['id', 'multi_1'])[target].transform(lambda x: x.rolling(roll_wind).median()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_1_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_median.pkl")
            else:
                df.groupby(['id', 'multi_1'])[target].transform(lambda x: x.rolling(roll_wind).mean()).to_pickle(jsn["FEATURE_LAG_DIR"] + f"multi_1_sales_residual_diff_28_roll_365_shift_{shift_day}_roll_{roll_wind}_mean.pkl")
        
        df = df.drop(target, axis=1)
    
    
    