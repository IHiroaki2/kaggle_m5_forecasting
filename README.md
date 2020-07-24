Hello!

Below you can find a outline of how to reproduce my solution for the m5-forecasting-uncertainty competition.
If you run into any trouble with the setup/code or have any questions please contact me at mtcra80509@yahoo.co.jp

## HARDWARE: 
"The following specs were used to create the original solution"

* SanDisk SecureAccessV3.01 2TB

* MacBook Pro - 2.4 GHz 8coreIntel Core i9 - 64 GB 2667 MHz DDR4

## SOFTWARE :
"python packages are detailed separately in `requirements.txt`"

Python 3.7.6


## How to Train

a) expect this to run about 8 ~ 9days

b) trains all models from scratch

### DATA SETUP 

"assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed"

below are the shell commands used in each step, as run from the top level directory

```zsh
cd Accuracy_uncertainty_prediction/m5-forecasting-accuracy/
kaggle competitions download -c m5-forecasting-accuracy -f sales_train_evaluation.csv
kaggle competitions download -c m5-forecasting-accuracy -f sell_prices.csv
kaggle competitions download -c m5-forecasting-accuracy -f calendar.csv
kaggle competitions download -c m5-forecasting-accuracy -f sample_submission.csv
cd ..
cd m5-forecasting-uncertainty
kaggle competitions download -c m5-forecasting-uncertainty -f sample_submission.csv
cd ..
```


### DATA PROCESSING

The train/predict code will also call this script if it has not already been run on the relevant data.

Make features, weights, dataframe to train and format to predict

```zsh
python prepare_data.py
```

### TRAIN MODELS

```zsh
python train_validation.py
python train_private.py
```


### ACCURACY PREDICTION

```zsh
python prediction_validation.py
python prediction_private.py "pred_acc_1.csv"   
```

>argument[1]:File name to store the predicted accuracy

### BLEND ACCURACY

If you want to blend, you can run this command.

```zsh
python blend_accuracy.py "blend_pred_acc_1_2.csv" "pred_acc_1.csv" "pred_acc_2.csv" ....... 
```

>argument[1]: File name to store the blended accuracy

>argument[2:] :  The accuracy files to blend(more than 2 files)

### UNCERTAINTY PREDICTION

```zsh
python uncertainty_predict_full.py "blend_pred_acc_1_2.csv" "uncertainty_submission_1.csv"
```

>argument[1]: The accuracy file used to predict uncertainty

>argument[2]: File name to store the predicted uncertainty

## Attention

Accuracy_uncertainty_prediction/accuracy_pred_shift_summary, dataframe, features, model, prediction, weights

The data in these folders will be overwritten, so if you have data you want to keep, please move it to another folder.

