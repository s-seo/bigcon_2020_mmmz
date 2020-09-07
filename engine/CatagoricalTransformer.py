import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

## input : X_train, X_test
## output : preprocessed X_train, X_test
## full pipeline에 모델 붙이면 y_train, y_test 구할 수 있음



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



## create catagorical transformer
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self):
        ## load data
        self.train = pd.read_csv("../data/00/2019sales.csv", skiprows = 1)
        self.train.rename(columns={' 취급액 ': '취급액'}, inplace = True)
        self.train['exposed']  = self.train['노출(분)']
        # define data types
        self.train.마더코드 = self.train.마더코드.astype(int).astype(str).str.zfill(6)
        self.train.상품코드 = self.train.상품코드.astype(int).astype(str).str.zfill(6)
        self.train.취급액 = self.train.취급액.str.replace(",","").astype(float)
        self.train.판매단가 = self.train.판매단가.str.replace(",","").replace(' - ', np.nan).astype(float)
        self.train.방송일시 = pd.to_datetime(self.train.방송일시, format="%Y/%m/%d %H:%M")
        self.train.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace = True)
        self.train['ymd'] = [d.date() for d in self.train["방송일시"]]
        self.train['volume'] = self.train['취급액'] / self.train['판매단가']
        # define ts_schedule, one row for each timeslot
        self.ts_schedule = self.train.copy().groupby('방송일시').first()
        self.ts_schedule.reset_index(inplace = True)

    # Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self

    # Helper function to extract year from column 'dates'
    # 여기서부터 파생변수 함수 짜면
    def function_name(self, obj):
        return obj

    # Transformer method we wrote for this transformer
    # 앞에서 정의한 함수를 이용하여 df transform하기
    # ***example code***

    def transform(self, X , y = None ):
       # Depending on constructor argument break dates column into specified units
       # using the helper functions written above
       for spec in self._use_dates:

        exec( "X.loc[:,'{}'] = X['date'].apply(self.get_{})".format( spec, spec ) )
       # Drop unusable column
       X = X.drop('date', axis = 1 )

       # Convert these columns to binary for one-hot-encoding later
       X.loc[:,'waterfront'] = X['waterfront'].apply( self.create_binary )

       X.loc[:,'view'] = X['view'].apply( self.create_binary )

       X.loc[:,'yr_renovated'] = X['yr_renovated'].apply( self.create_binary )
       # returns numpy array
       return X.values





## unite Features

# Categrical features to pass down the categorical pipeline
categorical_features = ['small_c',... ]

# Numerical features to pass down the numerical pipeline
numerical_features = ['sales_power',...]

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
full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ),

                                                  ( 'numerical_pipeline', numerical_pipeline ) ] )
