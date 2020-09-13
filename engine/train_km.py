#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General imports
import numpy as np
import pandas as pd
import sys, gc, time, warnings, pickle, random, psutil

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

import warnings
warnings.filterwarnings('ignore')

import os

import tqdm
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler #StandardScaler

?? Model layers 고쳐야함 ㅜ ??
from keras_models import create_model17EN1EN2emb1, create_model17noEN1EN2, create_model17

import json
#with open('SETTINGS.json', 'r') as myfile:
#    datafile=myfile.read()


?? SETTINGS 새로 짜서 저장경로 만들기 ??
SETTINGS = json.loads(datafile)
data_path = SETTINGS['RAW_DATA_DIR']
PROCESSED_DATA_DIR = SETTINGS['PROCESSED_DATA_DIR']
MODELS_DIR = SETTINGS['MODELS_DIR']
LGBM_DATASETS_DIR = SETTINGS['LGBM_DATASETS_DIR']

###############################################################################
################################# LIGHTGBM ####################################
###############################################################################



########################### Helpers
###############################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


?? Predict.py 에 ROLS_split이랑 같이 쓰임 ??
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


########################### Helper to load data by weekend/weekday
###############################################################################

#### Load Data#############
############################################################################
?? Train / Test / Val. 나누기 ############################################
############################################################################

## WeekEnd(we) & WeekDay(wd) & YES lag
train_wd_lag = pd.read_csv()
train_we_lag = pd.read_csv()
val_wd_lag = pd.read_csv()
val_we_lag = pd.read_csv()
test_wd_lag = pd.read_csv()
test_we_lag = pd.read_csv()

## WeekEnd(we) & WeekDay(wd) & NO lag
train_wd_nolag = pd.read_csv()
train_we_nolag = pd.read_csv()
val_wd_nolag = pd.read_csv()
val_we_nolag = pd.read_csv()
test_wd_nolag = pd.read_csv()
test_we_nolag = pd.read_csv()

## Nothing & YES lag
train_full_lag = pd.read_csv()
val_full_lag = pd.read_csv()
test_full_lag = pd.read_csv()

## Nothing & NO lag
train_full_nolag = pd.read_csv()
val_full_nolag = pd.read_csv()
test_full_nolag = pd.read_csv()


lag_features=[
       #########################################
       ??    LAG / RM / RS should be put    ###
       #########################################
    ]




########################### Model params
###############################################################################
import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.6,
                    'subsample_freq': 1,
                    'learning_rate': 0.02,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.6,
                    'max_bin': 100,
                    'n_estimators': 1600,
                    'boost_from_average': False,
                    'verbose': -1,
                    'num_threads': 12
                } 


########################### Baseline model
#################################################################################

# We will need some global VARS for future

SEED = 42             # Our random seed for everything
random.seed(SEED)     # to make all tests "deterministic"
np.random.seed(SEED)

TARGET = '취급액'      # Our Target


# Our baseline model serves
# to do fast checks of
# new features performance 


# We will use LightGBM for our tests
import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',         # Standart boosting type
                    'objective': 'regression',       # Standart loss for RMSE
                    'metric': ['mape'],              # as we will use rmse as metric "proxy"
                    'subsample': 0.8,                
                    'subsample_freq': 1,
                    'learning_rate': 0.05,           # 0.5 is "fast enough" for us
                    'num_leaves': 2**7-1,            # We will need model only for fast check
                    'min_data_in_leaf': 2**8-1,      # So we want it to train faster even with drop in generalization 
                    'feature_fraction': 0.8,
                    'n_estimators': 5000,            # We don't want to limit training (you can change 5000 to any big enough number)
                    'early_stopping_rounds': 30,     # We will stop training almost immediately (if it stops improving) 
                    'seed': SEED,
                    'verbose': -1,
                } 

## MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Small function to make fast features tests
# estimator = make_fast_test(grid_df)
# it will return lgb booster for future analisys
def make_fast_test(df):

    #features_columns = [col for col in list(df) if col not in remove_features]

    #tr_x, tr_y = df[df['d']<=(END_TRAIN-28)][features_columns], df[df['d']<=(END_TRAIN-28)][TARGET]              
    #vl_x, v_y = df[df['d']>(END_TRAIN-28)][features_columns], df[df['d']>(END_TRAIN-28)][TARGET]
    
    train_data = lgb.Dataset(??tr_x, label=??tr_y)
    valid_data = lgb.Dataset(??vl_x, label=??v_y)
    
    estimator = lgb.train(
                            lgb_params,
                            train_data,
                            valid_sets = [train_data,valid_data],
                            verbose_eval = 500,
                        )
    
    return estimator

# Make baseline model
baseline_model = make_fast_test(?? grid_df)


########################### Vars
###############################################################################

# Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
lgb_params['seed'] = SEED        # as possible
#N_CORES = psutil.cpu_count()     # Available CPU cores
#N_CORES=7 #from 24


#LIMITS and const
TARGET      = 'sales'            # Our target
#P_HORIZON   = 28                 # Prediction horizon
#USE_AUX     = False               # Use or not pretrained models

#FEATURES to remove
remove_features = [ ]


#PATHS for Features
ORIGINAL = data_path#
BASE     = PROCESSED_DATA_DIR+'grid_part_1_eval.pkl'
LAGS     = PROCESSED_DATA_DIR+'lags_df_28_eval.pkl'


# AUX(pretrained) Models paths
#AUX_MODELS = data_path+'m5-aux-models/'


#SPLITS for lags creation
??SHIFT_DAY  = 28
??N_LAGS     = 15
??LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
??ROLS_SPLIT = []
??for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])
        

#N_CORES=7 #from 24


#dataPath = '/var/data/m5-forecasting-accuracy/'
# train_df = pd.read_csv(dataPath+'sales_train_validation.csv')
# train_df = pd.read_csv(data_path+'sales_train_evaluation.csv')



features_columns = [  ]



?? rounds_per_model={
     'CA_1': 700,
     'CA_2': 1100,
     'CA_3': 1600,
     'CA_4': 1500,
     'TX_1': 1000, 
     'TX_2': 1000,
     'TX_3': 1000,
     'WI_1': 1600,
     'WI_2': 1500,
     'WI_3': 1100
}
#rounds_per_store1={
#     'CA_1': 1,
#     'CA_2': 1,
#     'CA_3': 1,
#     'CA_4': 1,
#     'TX_1': 1,
#     'TX_2': 1,
#     'TX_3': 1,
#     'WI_1': 1,
#     'WI_2': 1,
#     'WI_3': 1
#}

###############################################################################
?? 추가필요 ##################### Hyperparameter Tuning #######################
###############################################################################



########################### Train Models
###############################################################################
#for store_id in STORES_IDS:
   # print('Train', store_id)
    lgb_params['n_estimators'] = rounds_per_model[??]  ?? for문 말고 다른걸로~~
    
    
    # Launch seeder again to make lgb training 100% deterministic
    # with each "code line" np.random "evolves" 
    # so we need (may want) to "reset" it
    seed_everything(SEED)
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )

    # Save model - it's not real '.bin' but a pickle file
    # estimator = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    # pickle.dump gives us more flexibility
    # like estimator.predict(TEST, num_iteration=100)
    # num_iteration - number of iteration want to predict with, 
    # NULL or <= 0 means use best iteration

#     model_name = 'lgbweights/lgb_model_'+store_id+'END_TRAIN'+str(END_TRAIN)+'_v'+str(VER)+'.bin'
    model_name = MODELS_DIR+'lgbm_finalmodel_'+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))

    # Remove temporary files and objects 
    # to free some hdd space and ram memory
#    !rm lgbtrainings/train_data.bin
  

# "Keep" models features for predictions
MODEL_FEATURES = features_columns     ?? Predict.py에서 쓰임

with open(LGBM_DATASETS_DIR+'lgbm_features.txt', 'w') as filehandle:
    for listitem in MODEL_FEATURES:
        filehandle.write('%s\n' % listitem)


###############################################################################
################################# KERAS #######################################
###############################################################################


# Read raw data
sales = pd.read_csv(data_path+ "sales_train_evaluation.csv")
# sales = pd.read_csv(data_path+"sales_train_validation.csv")



cat_cols = [??]
num_cols = [ ]
bool_cols = [ ]     ## Dummy
dense_cols = num_cols + bool_cols


?? LOG Transformation 할건가????????
sales['logd']=np.log1p(sales.d-sales.d.min())


# Need to do column by column due to memory constraints
for i, v in tqdm.tqdm(enumerate(num_cols)):
    sales[v] = sales[v].fillna(sales[v].median())##############################
    

gc.collect()


# Input dict for training with a dense array and separate inputs for each embedding input
def make_X(df):
    X = {"dense1": df[dense_cols].values}
#     X = {"dense1": df[dense_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
#         X[v] = df[[v]].to_numpy()
        X[v] = df[[v]].values
    return X


#END_TRAIN=1941
#et=1941


base_epochs=30
ver='EN1EN2Emb1'


flag = (sales.d < et+1 )& (sales.d > et+1-17*28 )

X_train = make_X(sales[flag])
y_train = sales["demand"][flag].values

X_train['dense1'], scaler1 = preprocess_data(X_train['dense1'])

val_long = sales[(sales.d >= et+1)].copy()

#train
model = create_model17EN1EN2emb1(num_dense_features=len(dense_cols), lr=0.0001)

history = model.fit(X_train,  y_train,
                        batch_size=2 ** 14,
                        epochs=base_epochs,
                        shuffle=True,
                        verbose = 2
                   )

model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')     
#import matplotlib.pyplot as plt 
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('plt_hist'+str(et)+ver)
#plt.show()

for j in range(4):        
    model.fit(X_train, y_train,
                        batch_size=2 ** 14,
                        epochs=1,
                        shuffle=True,
                        verbose = 2
             )
    model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+1+j)+'_ver-'+ver+'.h5')          



# Second Model
base_epochs=30
ver='noEN1EN2'

#train
model = create_model17noEN1EN2(num_dense_features=len(dense_cols), lr=0.0001)
history = model.fit(X_train,  y_train,
                        batch_size=2 ** 14,
                        epochs=base_epochs,
                        shuffle=True,
                        verbose = 2
                   )

model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')      
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('plt_hist'+str(et)+ver)
#plt.show()
for j in range(4):        
    model.fit(X_train, y_train,
                        batch_size=2 ** 14,
                        epochs=1,
                        shuffle=True,
                        verbose = 2
             )
    model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+1+j)+'_ver-'+ver+'.h5')          

# del model
gc.collect()

# Third Model
base_epochs=30
ver='17last'

val_long = sales[(sales.d >= et+1)].copy()

#train
model = create_model17(num_dense_features=len(dense_cols), lr=0.0001)
history = model.fit(X_train,  y_train,
                        batch_size=2 ** 14,
                        epochs=base_epochs,
                        shuffle=True,
                        verbose = 2
                   )

model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs)+'_ver-'+ver+'.h5')      
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.savefig('plt_hist'+str(et)+ver)
#plt.show()
for j in range(4):        
    model.fit(X_train, y_train,
                        batch_size=2 ** 14,
                        epochs=1,
                        shuffle=True,
                        verbose = 2
             )
    model.save_weights(MODELS_DIR+'Keras_CatEmb_final3_et'+str(et)+'ep'+str(base_epochs+1+j)+'_ver-'+ver+'.h5')          


# del model
gc.collect()        