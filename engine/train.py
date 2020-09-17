# General imports
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import random
import sys, gc, time
import os

# data
import datetime
import itertools
import json
import pickle

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


# model
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# custom modules
# from engine.features_yj import Features
from engine.preprocess import load_df_added, drop_useless, check_na, run_label_all, remove_outliers, run_stdscale
import tensorflow as tf
import tensorflow.keras as keras


###############################################################################
################################# Load Data ####################################
###############################################################################

## Set Directories
## Data is NOT RAW and has all features

local_DIR = os.getcwd()
featured_DATA_DIR = local_DIR + '/data/20'
# ?PROCESSED_DATA_DIR = local_DIR +'/data/21'

## Import 6 types of dataset
## Descriptions:
#   - df_wd_lag : weekday / + lags
#   - df_wd_no_lag : weekday
#   - df_wk_lag: weekend / + lags
#   - df_wk_no_lag : weekend
#   - df_all_lag : all days / +lags
#   - df_all : all days

df_wd_lag = pd.read_pickle(featured_DATA_DIR + '/train_fin_wd_lag.pkl').reset_index()
df_wd_no_lag = pd.read_pickle(featured_DATA_DIR + '/train_fin_wd_no_lag.pkl').reset_index()
df_wk_lag = pd.read_pickle(featured_DATA_DIR + '/train_fin_wk_lag.pkl').reset_index()
df_wk_no_lag = pd.read_pickle(featured_DATA_DIR + '/train_fin_wk_no_lag.pkl').reset_index()
df_all_lag = pd.read_pickle(featured_DATA_DIR + '/train_fin_light_ver.pkl').reset_index()
#df_all = pd.read_pickle(featured_DATA_DIR + '/train_fin_wk_lag.pkl')


## drop unnecessary lag columns
df_wd_lag = df_wd_lag.drop(columns = ['lag_sales_wk_1','lag_sales_wk_2'])
df_wk_lag = df_wk_lag.drop(columns = ['lag_sales_wd_1', 'lag_sales_wd_2','lag_sales_wd_3', 'lag_sales_wd_4', 'lag_sales_wd_5'])

df_wk_lag.shape # (10060, 89)
df_wk_no_lag.shape # (10060, 83)
df_wd_lag.shape # (25319, 92)
df_wd_no_lag.shape # (25319, 83)
df_all_lag.shape # (35379, 91)


## set some Global Vars
data_list = ['df_wk_lag','df_wk_no_lag','df_wd_lag','df_wd_no_lag','df_all_lag']

lag_col1 = ['lag_scode_count','lag_mcode_price','lag_mcode_count','lag_bigcat_price','lag_bigcat_count',
            'lag_bigcat_price_day','lag_bigcat_count_day','lag_small_c_price','lag_small_c_count']

lag_col2 = ['rolling_mean_7', 'rolling_mean_14', 'lag_sales_wd_1', 'lag_sales_wd_2','lag_sales_wd_3',
            'lag_sales_wd_4', 'lag_sales_wd_5', 'lag_sales_wk_1','lag_sales_wk_2', 'ts_pred',
           'rolling_mean_mcode_7','rolling_mean_mcode_14',]

num_col = ['']

cat_col = ['상품군','weekdays','show_id','small_c','middle_c','big_c',
                        'pay','months','hours_inweek','weekends','japp','parttime',
                        'min_start','primetime','prime_origin','prime_smallc',
                        'freq','bpower','steady','men','pay','luxury',
                        'spring','summer','fall','winter','rain']
len(lag_col1)
len(cat_col)




###############################################################################
############################### Preprocess  ##################################
###############################################################################


check_na(df_wk_lag.iloc[:,:40])

## simple function that will be used for run_preprocess
def na_to_zeroes(df):
    """
    :objective: Change all na's to zero.(just for original lag!)
    :return: pandas dataframe
    """
    xcol = [x for x in df.columns if x in lag_col1+lag_col2+['mid_click_r']]
    for col in xcol:
        df[col] = df[col].fillna(0)

    return df

def drop_cat(df_pca):
    """
    :objective: Before PCA, drop categorical variables
    :return: pandas dataframe
    """
    xcol = [x for x in df_pca.columns if x in cat_col+lag_col2]
    df_pca = df_pca.drop(columns = xcol)
    df_pca = df_pca.drop(columns = '취급액')

    return df_pca

def run_pca(df_pca_scaled, n_components = 5):
    """
    :objective: Run PCA with n_components = 5
    :return: pandas dataframe
    """
    pca = PCA(n_components = 5)
    pca.fit(df_pca_scaled)
    df_pca = pca.transform(df_pca_scaled)

    return df_pca

## run preprocessing in a shot
## pca is optional and only applied to numeric features other than 'lag'
## NOTICE: removing outliers were run prior to dividing train/val
## if replace = True, new PCA will replace corresponding numerical columns
## if you want to simply add PCA columns to original data, set replace = False
def run_preprocess(df, pca = True, replace = True):
    """
    :objective: Run Feature deletion, NA imputation, label encoding, pca(optional)
    :return: pandas dataframe
    """
    df = drop_useless(df)
    df = na_to_zeroes(df)
    df = remove_outliers(df)
    df = run_label_all(df)
    df1 = df.copy()
    if pca:
        xcol = [x for x in df1.columns if x in cat_col+lag_col2]
        df_pca = df1.copy()
        df_pca = drop_cat(df_pca).copy()
        df_pca = run_stdscale(df_pca)
        df_pca = run_pca(df_pca)
        if replace:
            df_pca1 = pd.concat([df1[xcol], pd.DataFrame(df_pca)], axis=1)
            return df_pca1
        else:
            df_pca2 = pd.concat([df1, pd.DataFrame(df_pca)], axis=1)
            return df_pca2
    else:
        return df1

## Preprocessed datasets
df_wk_lag_PP = run_preprocess(df_wk_lag, pca = False, replace =False)
df_wk_no_lag_PP = run_preprocess(df_wk_no_lag, pca = False, replace =False)
df_wd_lag_PP = run_preprocess(df_wd_lag, pca = False, replace = False)
df_wd_no_lag_PP = run_preprocess(df_wd_no_lag, pca = False, replace =False)
df_all_lag_pp = run_preprocess(df_all_lag, pca = False, replace =True)

#df_wd_lag_PP.to_csv ('df_wd_lag_PP.csv', index = False, header=True,encoding='ms949')
#df_wd_no_lag_PP.to_csv ('df_wd_no_lag_PP.csv', index = False, header=True,encoding='ms949')




###############################################################################
############################### Helper Functions  ##################################
###############################################################################

## quickly check distribution
import seaborn as sns
plt.figure(figsize=(10,10))
sns.distplot(df_wd_lag_PP.취급액)
sns.distplot(df_wd_lag_PP.rolling_mean_14)
df_wd_lag_PP.columns


## Seeder
def seed_everything(seed=127):
    random.seed(seed)
    np.random.seed(seed)

## metrics
# negative mape
def neg_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    result = (-1)*mape
    return result

# RMSE
def get_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse

# MAE
def get_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


## CV splits
def cv_split(df, month, printprop = False):
    split = int(df[df['months']==month].index.values.max())
    prop = str(split/df.shape[0])
    if printprop:
        print(f'Proportion of train set is {prop}')
        return split
    else:
        return split


## Divide into train/test
def divide_train_val(df_pp, month, drop):
    split = cv_split(df = df_pp, month = month)
    train_x = df_wd_lag_PP.iloc[:split,:].drop(columns = ['index','show_id','취급액']+drop)
    train_y = df_wd_lag_PP.iloc[:split,:].취급액
    val_x = df_wd_lag_PP.iloc[split:,:].drop(columns = ['index','show_id','취급액']+drop)
    val_y = df_wd_lag_PP.iloc[split:,:].취급액
    return train_x, train_y, val_x, val_y


train_x, train_y, val_x, val_y = divide_train_val(df_wd_lag_PP, 8, drop = ['small_c'])
train_x.shape





###############################################################################
########################### Light GBM ##############################
###############################################################################



TARGET = '취급액'      # Our Target

gbm = LGBMRegressor(objective = 'regression',
                     boosting_type = 'dart',
                     metric = 'mape',
                     n_estimators = 3000, #10000
                     num_leaves = 10, #10
                     learning_rate = 0.002, #0.01
                     bagging_fraction = 0.9,
                     feature_fraction = 0.5,
                     bagging_seed = 0,
                     categorical_feature = [0,9,10,11,3],
                     #max_depth = 10,
                                         )

def run_lgbm(train_x, train_y, val_x, val_y):
    seed_everything(seed=127)

    estimator = gbm.fit(train_x,train_y,
                        eval_set=[(val_x, val_y)],
                        verbose = 100,
                        eval_metric = 'mape',
                        early_stopping_rounds = 100
                        )
    lgbm_preds = gbm.predict(val_x, num_iteration= estimator.best_iteration_)
    lgbm_preds[lgbm_preds < 0] = 0

    # plot
    x = range(0,val_y.shape[0])
    plt.figure(figsize=(50,10))
    plt.plot(x,val_y,label='true')
    plt.plot(x,lgbm_preds, label='predicted')
    plt.legend()

    plt.show()

    # show scores
    print(f'MAPE of best iter is {neg_mape(val_y,lgbm_preds)}')
    print(f'MAE of best iter is {get_mae(val_y,lgbm_preds)}')
    print(f'RMSE of best iter is {get_rmse(val_y,lgbm_preds)}')

    # feature Importance
    #fi = {'name' : estimator.feature_name_,'importance': estimator.feature_importances_}
    #fi = pd.DataFrame(fi, columns = ['name','importance'])
    #fi.sort_values(by=['importance'], inplace=True, ascending = False)

    #return fi

# save model
data_type = 'wk_lag'
model_name = MODELS_DIR+'lgbm_finalmodel_'+data_type+'.bin'
pickle.dump(estimator, open(model_name, 'wb'))



def make_fast_test(train_x, train_y, val_x, val_y):

    train_data = lgb.Dataset(train_x, label=train_y)
    valid_data = lgb.Dataset(val_x, label=val_y)

    estimator = lgb.train(
                            lgb_params,
                            train_data,
                            valid_sets = [train_data,valid_data],
                            verbose_eval = 500,
                        )

    return estimator

# Make baseline model
baseline_model = make_fast_test(train_x, train_y, val_x, val_y)


# run for each cv
cv_months = [7,8,9]

for num in cv_months:
    month = num
    train_x, train_y, val_x, val_y = divide_train_val(df_wd_lag_PP, month, drop = ['small_c'])
    print(f'CV with month {month} is starting.')
    run_lgbm(train_x, train_y, val_x, val_y)









########################### Hyperparameter Tuning
###############################################################################

## Score metric
my_scorer = make_scorer(neg_mape, greater_is_better=True)


## Optimization objective
hp_d = {}

#cv3 train
#1~8
#1~7
#1~9

#robust cv : 1~8로 12월 예측하기

def cv_score(learning_rate, subsample, num_leaves, n_estimators, num_iterations, feature_fraction, min_data_in_leaf,
             scale_pos_weight):
    hp_d['learning_rate'] = learning_rate
    hp_d['subsample'] = subsample
    hp_d['n_estimators'] = int(n_estimators)
    hp_d['num_iterations'] = int(num_iterations)
    hp_d['num_leaves'] = int(num_leaves)
    hp_d['feature_fraction'] = feature_fraction
    hp_d['min_data_in_leaf'] = int(min_data_in_leaf)
    hp_d['scale_pos_weight'] = scale_pos_weight

    score = cross_val_score(
                lgb.LGBMRegressor(boosting_type= 'gbdt', objective= 'regression', metric= ['mape'], seed=42,
                                  subsample_freq=1, max_depth=-1,
                                  categorical_feature= [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], **hp_d),
                train_x, train_y, cv=5, scoring=my_scorer).mean()
    return score


## Execution
from bayes_opt import BayesianOptimization

bayes_optimizer = BayesianOptimization(f=cv_score, pbounds={'learning_rate':(0,0.05),
                                                          'subsample':(0,1),
                                                          'n_estimators':(3000,5000),
                                                            'num_iterations':(1000,3000),
                                                          'num_leaves':(10,20),
                                                          'feature_fraction':(0.6,1),
                                                          'min_data_in_leaf':(100,300),
                                                           'scale_pos_weight': (1,2)},
                                      random_state=42, verbose=2)
bayes_optimizer.maximize(init_points=3, n_iter=47, acq='ei', xi=0.01)

## print all iterations
for i,res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
## print best iterations
print('Final result: ', bayes_optimizer.max)


## Final result로 돌려보기

#BO_result = {'feature_fraction': 0.7494834785935099, 'learning_rate': 0.008211209428118416, 'min_data_in_leaf': 298, 'n_estimators': 3304, 'num_iterations': 2671, 'num_leaves': 13, 'scale_pos_weight': 1.746019312719247, 'subsample': 0.7734458936226617}
#BO_result['boosting_type'] = 'gbdt'
#BO_result['objective'] = 'regression'
#BO_result['metric'] = 'mape'
#BO_result['subsample_freq'] = 1
#BO_result['categorical_feature'] = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

#BO_train = lgb.train(BO_result, train_data,
                           # valid_sets = [train_data,valid_data],
                           # verbose_eval = 500 )


def BO_execute(train_data, valid_data):
    bayes_optimizer = BayesianOptimization(f=cv_score, pbounds={'learning_rate':(0,0.05),
                                                          'subsample':(0,1),
                                                          'n_estimators':(3000,5000),
                                                            'num_iterations':(1000,3000),
                                                          'num_leaves':(10,20),
                                                          'feature_fraction':(0.7,1),
                                                          'min_data_in_leaf':(100,300),
                                                           'scale_pos_weight': (1,2)},
                                      random_state=42, verbose=2)
    bayes_optimizer.maximize(init_points=3, n_iter=47, acq='ei', xi=0.01)
    BO_best = bayes_optimizer.max['params']
    BO_best['boosting_type'] = 'gbdt'              ## 얘네 말고도 걍 고정시킬 하이퍼파라미터들은 여기에 때려넣기
    BO_best['objective'] = 'regression'
    BO_best['metric'] = 'mape'
    ## BO_best['categorical_feature'] = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

    BO_result = lgb.train(BO_best, train_data,
                            valid_sets = [train_data,valid_data],
                            verbose_eval = 500 )
    return BO_best, BO_result

BO_params, BO_model = BO_execute(train_data, valid_data)

## <TO DO LIST - Tuesday 10:00PM>
## cv 함수 다시 / 전처리 완료 데이터 주기
## lightgbm cat feature 처리 방법
## 언니: weekday/wd 나: weekend/wk 돌려보
## Feature Importance
## model 저장 pickle로




###############################################################################
########################### KERAS ##############################
###############################################################################
save_model_weights




###############################################################################
########################### Ensemble ##############################
###############################################################################
lgbm_preds
keras_preds

# 판매단가 기준
# grid search
def set_threshold:

for num in thresholds:


MAPE
mae


###############################################################################
########################### Predict ##############################
###############################################################################

submission = pd.Dataframe()
