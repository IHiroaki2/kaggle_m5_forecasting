#!/bin/sh

mkdir accuracy_pred_shift_summary
mkdir accuracy_submission
mkdir dataframe
mkdir features
mkdir features/Basic
mkdir features/Encoding
mkdir features/Lag_Features
mkdir features/Price
mkdir features/Target
mkdir m5-forecasting-accuracy
mkdir m5-forecasting-uncertainty
mkdir model
mkdir prediction
mkdir prediction/format
mkdir uncertainty_submission
mkdir weights


cd m5-forecasting-accuracy
kaggle competitions download -c m5-forecasting-accuracy
cd ..


cd m5-forecasting-uncertainty
kaggle competitions download -c m5-forecasting-uncertainty
cd ..


unzip m5-forecasting-accuracy/m5-forecasting-accuracy.zip -d m5-forecasting-accuracy
unzip m5-forecasting-uncertainty/m5-forecasting-uncertainty.zip -d m5-forecasting-uncertainty