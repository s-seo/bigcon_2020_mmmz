# General imports
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
import os

# data
import pickle

# sklearn
from sklearn.preprocessing import RobustScaler

# model
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from engine.preprocess import load_df, run_preprocess
from data.vars import *

###############################################################################
################################# Load Data ###################################
###############################################################################

## Set Directories
## Data is NOT RAW and has all features

local_DIR = os.getcwd()
MODELS_DIR = local_DIR + "/data/saved_models"
featured_DATA_DIR = local_DIR + '/data/fin_data'

## Import 6 types of dataset
## Descriptions:
#   - df_wd_lag : weekday / + lags
#   - df_wk_lag: weekend / + lags
#   - df_all_lag : all days / +lags

df_wd_lag = load_df(featured_DATA_DIR + '/train_fin_wd_lag.pkl')
df_wk_lag = load_df(featured_DATA_DIR + '/train_fin_wk_lag.pkl')
df_all_lag = load_df(featured_DATA_DIR + '/train_fin_light_ver.pkl')

###############################################################################
############################### Preprocess  ##################################
###############################################################################

## Preprocessed datasets
df_wk_lag_PP = run_preprocess(df_wk_lag)
df_wd_lag_PP = run_preprocess(df_wd_lag)

## Scaled Data
df_wd_lag_PP_s = df_wd_lag_PP.copy()
df_wk_lag_PP_s = df_wk_lag_PP.copy()

###############################################################################
############################### Helper Functions  #############################
###############################################################################

## Seeder
def seed_everything(seed=127):
    random.seed(seed)
    np.random.seed(seed)


## metrics
# negative mape (For Bayesian Optimization)
def neg_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    result = (-1) * mape
    return result

# MAPE
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    final = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return final

# RMSE
def get_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# MAE
def get_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

## CV splits
def cv_split(df, month, printprop=False):
    split = int(df[df['months'] == month].index.values.max())
    prop = str(split / df.shape[0])
    if printprop:
        print(f'Proportion of train set is {prop}')
        return split
    else:
        return split


## Divide into train/test
def divide_train_val(df_pp, month, drop):
    split = cv_split(df=df_pp, month=month)
    train_x = df_pp.iloc[:split, :].drop(columns=['index',
                                                  'show_id', '취급액'] + drop)  ## 'index' 다시 한 번 check 필요!!!!!!!!
    train_y = df_pp.iloc[:split, :].취급액
    val_x = df_pp.iloc[split:, :].drop(columns=['index',
                                                'show_id', '취급액'] + drop)
    val_y = df_pp.iloc[split:, :].취급액
    return train_x, train_y, val_x, val_y


## set month
## WD
train_wd_lag_x_s, train_wd_lag_y_s, val_wd_lag_x_s, val_wd_lag_y_s = divide_train_val(df_wd_lag_PP_s, 8, drop=[])
train_wd_lag_x, train_wd_lag_y, val_wd_lag_x, val_wd_lag_y = divide_train_val(df_wd_lag_PP, 8, drop=[])

## WK
train_wk_lag_x_s, train_wk_lag_y_s, val_wk_lag_x_s, val_wk_lag_y_s = divide_train_val(df_wk_lag_PP_s, 8, drop=[])
train_wk_lag_x, train_wk_lag_y, val_wk_lag_x, val_wk_lag_y = divide_train_val(df_wk_lag_PP, 8, drop=[])

###############################################################################
########################### Light GBM ##############################
###############################################################################


TARGET = '취급액'  # Our Target

params_lg_wd = {'feature_fraction': 1,
                'learning_rate': 0.001,
                'min_data_in_leaf': 135,
                'n_estimators': 3527,
                'num_iterations': 2940,
                'subsample': 1,
                'boosting_type': 'dart',
                'objective': 'regression',
                'metric': 'mape',
                'categorical_feature': [3, 9, 10, 11]  ## weekdays, small_c, middle_c, big_c
                }

params_lg_wk = {'feature_fraction': 1,
                'learning_rate': 0.001,
                'min_data_in_leaf': 134,
                'n_estimators': 3474,
                'num_iterations': 2928,
                'subsample': 1,
                'boosting_type': 'dart',
                'objective': 'regression',
                'metric': 'mape',
                'categorical_feature': [3, 9, 10, 11]}  ## weekdays, small_c, middle_c, big_c


def run_lgbm(params, train_x, train_y, val_x, val_y):
    seed_everything(seed=127)

    global model_lg

    model_lg = LGBMRegressor(**params)
    model_lg.fit(train_x, train_y)
    lgbm_preds = model_lg.predict(val_x)

    ## Plot LGBM: Predicted vs. True values
    plt.figure(figsize=(20, 5), dpi=80)
    x = range(0, len(lgbm_preds))
    plt.plot(x, val_y, label='true')
    plt.plot(x, lgbm_preds, label='predicted')
    plt.legend()
    plt.title('LGBM')
    plt.show()

    ## Get Scores
    print(f'MAPE of best iter is {get_mape(val_y, lgbm_preds)}')
    print(f'MAE of best iter is {get_mae(val_y, lgbm_preds)}')
    print(f'RMSE of best iter is {get_rmse(val_y, lgbm_preds)}')


# Run & Save model
## WD
# run_lgbm(params_lg_wd, train_wd_lag_x, train_wd_lag_y, val_wd_lag_x, val_wd_lag_y)
data_type = 'wd_lag'
model_name = MODELS_DIR + 'lgbm_finalmodel_' + data_type + '.bin'
pickle.dump(model_lg, open(model_name, 'wb'))

## WK
# run_lgbm(params_lg_wk, train_wk_lag_x, train_wk_lag_y, val_wk_lag_x, val_wk_lag_y)
data_type = 'wk_lag'
model_name = MODELS_DIR + 'lgbm_finalmodel_' + data_type + '.bin'
pickle.dump(model_lg, open(model_name, 'wb'))


#############################################################################
########################### Hyperparameter Tuning ###########################
#############################################################################


# robust cv : 1~8로 12월 예측하기

'''
class BO:
    def __init__(self, df, tr_x, tr_y, v_x, v_y):
        self.data = df

    def neg_mape(self, y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        result = (-1)*mape
        return result

    def cv_score(self, learning_rate, num_iterations, n_estimators, min_data_in_leaf, feature_fraction, subsample):
        hp_d = {}
        hp_d['learning_rate'] = learning_rate   
        hp_d['n_estimators'] = int(n_estimators)
        hp_d['num_iterations'] = int(num_iterations)
        hp_d['subsample'] = subsample
        #hp_d['num_leaves'] = int(num_leaves)    
        hp_d['feature_fraction'] = feature_fraction
        hp_d['min_data_in_leaf'] = int(min_data_in_leaf)
        #hp_d['scale_pos_weight'] = scale_pos_weight

        data_cv = self.data

        tscv = TimeSeriesSplit(n_splits = 3)
        score = []
        for train_index, val_index in tscv.split(data_cv):
            cv_train, cv_val = data_cv.iloc[train_index], data_cv.iloc[val_index]
            cv_train_x = cv_train.drop('취급액', axis=1)
            cv_train_y = cv_train.취급액
            cv_val_x = cv_val.drop('취급액', axis=1)
            cv_val_y = cv_val.취급액

            model = lgb.LGBMRegressor(boosting_type= 'dart', objective= 'regression', metric= ['mape'], seed=42,
                                  subsample_freq=1, max_depth=-1,
                                  categorical_feature= [3,9,10,11], 
                                      **hp_d)

            model.fit(cv_train_x, cv_train_y)
            pred_values = model.predict(cv_val_x)
            true_values = cv_val_y

            score.append(self.neg_mape(true_values, pred_values))

        result  = np.mean(score)
        return result

    def BO_execute(self):

        int_list = ['n_estimators', 'num_iterations', 'min_data_in_leaf'
                   #,'num_leaves'
                   ]

        bayes_optimizer = BayesianOptimization(f=self.cv_score, pbounds={'learning_rate':(0.0001,0.01),
                                                          'subsample':(0.5,1),
                                                          'n_estimators':(3000,6000),  
                                                          'num_iterations':(2000,5000),
                                                          'feature_fraction':(0.6,1),
                                                          'min_data_in_leaf':(100,300),
                                                           #'scale_pos_weight': (1,2)
                                                                        },
                                      random_state=42, verbose=2)

        bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)

        ## print all iterations
        #for i,res in enumerate(bayes_optimizer.res):
        #    print('Iteration {}: \n\t{}'.format(i, res))
        ## print best iterations
        #print('Final result: ', bayes_optimizer.max)


        BO_best = bayes_optimizer.max['params'] 
        for i in int_list:
            BO_best[i] = int(np.round(BO_best[i]))
        BO_best['boosting_type'] = 'dart'             
        BO_best['objective'] = 'regression'
        BO_best['metric'] = 'mape'
        BO_best['categorical_feature'] = [3,9,10,11]
        #BO_best['feature_fraction'] = 1
        #BO_best['subsample'] = 1

        train_x = self.tr_x
        train_y = self.tr_y
        val_x = self.v_x
        val_y = self.v_y
        data_train = lgb.Dataset(train_x, label=train_y)
        data_valid = lgb.Dataset(val_x, label=val_y)

        BO_result = lgb.train(BO_best, data_train,
                            valid_sets = [data_train, data_valid],
                            verbose_eval = 500,
                             )
        return BO_best, BO_result

params_wd, model_wd = BO(df_wd_lag_PP, train_wd_lag_x, train_wd_lag_y, val_wd_lag_x, val_wd_lag_y).BO_execute()
params_wk, model_wk = BO(df_wk_lag_PP, train_wk_lag_x, train_wk_lag_y, val_wk_lag_x, val_wk_lag_y).BO_execute()
'''
