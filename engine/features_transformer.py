from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

from features_categorical import CategoricalTransformer
from features_numerical import NumericalTransformer


## create selector : column 이름에 따라 df의 column 선택해줌
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    #Return self nothing else to do here
    def fit( self, X, y = None ):
        return self

    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ]


## unite Features

# Categrical features to pass down the categorical pipeline
categorical_features = ['상품군','ymd','weekdays','show_id','브랜드','original_c','small_c',
                        'small_c_code','middle_c','middle_c_code','big_c','big_c_code',
                        'pay','months','days','hours','week_num','hours_inweek','japp','parttime',
                        'min_start','min_range','primetime']

# Numerical features to pass down the numerical pipeline
numerical_features = ['판매단가','exposed','volume','holidays','red','weekends','prime_origin',
                      'prime_smallc','sales_power','freq','dup_times','lag_scode_price','lag_scode_count',
                      'lag_mcode_price','lag_mcode_count','lag_bigcat_price','lag_bigcat_count','lag_bigcat_price_day',
                      'lag_bigcat_count_day','lag_small_c_price','lag_small_c_count','lag_all_price_show','lag_all_price_day',
                      'bpower','stead','men','luxury','vratings','sprin','summer','fall','winter','small_click_r',
                      'mid_click_r','big_click_r','rain','temp_diff_s'
                      ]

# Defining the steps in the categorical pipeline
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ),

                                  ( 'cat_transformer', CategoricalTransformer() ),

                                  ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] )

# Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ),

                                  ( 'num_transformer', NumericalTransformer() ),

                                  ('imputer', SimpleImputer(strategy = 'mean') ),

                                  ( 'std_scaler', StandardScaler() ) ] )



## merge into full Pipeline

# Combining numerical and categorical piepline into one full big pipeline horizontally
# using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),

                                                  ('numerical_pipeline', numerical_pipeline)])
