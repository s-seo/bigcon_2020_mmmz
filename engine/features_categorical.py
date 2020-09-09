import numpy as np
import pandas as pd

import datetime
import random
import math
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

## input : X_train, X_test
## output : preprocessed X_train, X_test
## full pipeline에 모델 붙이면 y_train, y_test 구할 수 있음


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

    ##################################
    ## onair time/order info variables
    ##################################

    def get_time(self, obj):
        """
        :** objective: get month, day, hours
        """
        self.train['months'] = self.train.방송일시.dt.month
        self.train['days'] = self.train.방송일시.dt.day
        self.train['hours'] = self.train.방송일시.dt.hour
        self.train['week_num'] = self.train.방송일시.dt.isocalendar()['week']

    def get_weekday(self, obj):
        """
        :** objective: get weekday
        """
        self.train['weekdays'] = self.train.방송일시.dt.day_name()

    def get_hours_inweek(self, obj):
        """
        :** objective: get hours by week (1~168)
        """
        hours_inweek = []
        for i in range(0, len(self.train)):
            hr = self.train['hours'].iloc[i]
            dy = self.train['weekdays'].iloc[i]
            if dy == 'Tuesday':
                hours_inweek.append(hr + 24)
            elif dy == 'Wednesday':
                hours_inweek.append(hr + 24 * 2)
            elif dy == 'Thursday':
                hours_inweek.append(hr + 24 * 3)
            elif dy == 'Friday':
                hours_inweek.append(hr + 24 * 4)
            elif dy == 'Saturday':
                hours_inweek.append(hr + 24 * 5)
            elif dy == 'Sunday':
                hours_inweek.append(hr + 24 * 6)
            else:
                hours_inweek.append(hr)
        self.train['hours_inweek'] = hours_inweek

    def get_min_start(self, obj):
        """
        :** objective: get startig time (min)
        """
        self.train['min_start'] = self.train.방송일시.dt.minute
        # list(set(train.방송일시.dt.minute)) #unique

    def filter_jappingt(self, obj):
        """
        :objective: round up 방송일시
        """
        japp = []
        for i in range(0, len(self.train)):
            time = self.train['방송일시'].iloc[i]
            if (time.minute < 30) & (time.hour == 0):
                rtn = time.hour
            elif time.minute >= 30:
                if time.hour == 23:
                    rtn = 0
                else:
                    rtn = time.hour + 1
            else:
                if time.hour == 0:
                    rtn = 23
                else:
                    rtn = time.hour
            japp.append(rtn)
        self.train['japp'] = japp

    def fill_exposed_na(self, obj):
        """
        :objective: fill out NA values on 'exposed' with mean(exposed)
        :return:pd.Dataframe with adjusted 'exposed' column
        """
        self.train["exposed"].fillna(self.train.groupby('방송일시')['exposed'].transform('mean'), inplace=True)

    def get_ymd(self, obj):
        """
        :objective: add 'ymd' variable to train dataset
        :return: pandas dataframe
        """
        t = 1
        while t < 9:
            for i in self.ts_schedule.ymd.unique():
                if i == datetime.date(2019, 1, 1): continue
                time_idx = self.ts_schedule[self.ts_schedule.ymd == i].index[0]
                first_show = self.ts_schedule.iloc[time_idx]
                last_show = self.ts_schedule.iloc[time_idx - 1]
                if (first_show['마더코드'] == last_show['마더코드']) & (
                        first_show['방송일시'] <= last_show['방송일시'] + datetime.timedelta(minutes=last_show['exposed'])):
                    self.ts_schedule.ymd.iloc[time_idx] = self.ts_schedule.ymd.iloc[time_idx - 1]

            t = t + 1

    def timeslot(self, obj):
        """
        :objective: get timeslot of each show
        """
        show_counts = [len(list(y)) for x, y in itertools.groupby(self.ts_schedule.상품코드)]  # count repeated 상품코드
        self.ts_schedule['parttime'] = ""  # define empty column
        j = 0
        for i in range(0, len(show_counts)):
            first_idx = j
            self.ts_schedule.parttime[first_idx] = 1
            j += show_counts[i]
            if show_counts[i] == 1:
                next
            self.ts_schedule.parttime[(first_idx + 1):j] = np.arange(2, show_counts[i] + 1)

        self.train['parttime'] = ""  # define empty column
        # add timeslot variable to train dataset
        for i in range(0, len(self.ts_schedule)):
            self.train.parttime[self.train.방송일시 == self.ts_schedule.방송일시[i]] = self.ts_schedule.parttime[i]

    def get_show_id(self, obj):
        """
        :objective: get show id for each day
        :return: pandas dataframe
        """
        self.ts_schedule['show_counts'] = ""
        for i in self.ts_schedule.ymd.unique():
            rtn = self.ts_schedule[self.ts_schedule.ymd == i]
            slot_count = 0  # number of shows for each day
            for j in range(0, len(rtn)):
                if rtn['parttime'].iloc[j] == 1:
                    slot_count += 1
                    idx = self.ts_schedule[self.ts_schedule.ymd == i].index[j]
                    self.ts_schedule.show_counts.iloc[idx] = str(i) + " " + str(slot_count)

    def get_min_range(self, obj):
        """
        :objective: get minutes aired for each show
        :return: pandas dataframe
        """
        self.ts_schedule['min_range'] = ""
        for i in range(0, len(self.ts_schedule)):
            if self.ts_schedule.parttime.iloc[i] == 1:
                min_dur = self.ts_schedule.exposed.iloc[i]
                j = i + 1
                if j == (len(self.ts_schedule)): break
                while self.ts_schedule.parttime.iloc[j] != 1:
                    min_dur += self.ts_schedule.exposed.iloc[j]
                    j += 1
                    if j == (len(self.ts_schedule)): break
            self.ts_schedule.min_range.iloc[i:j] = min_dur

    def add_showid_minran_to_train(self, obj):
        """
        :objective: add show_id and min_range column to train data
        :return: pandas dataframe
        """
        self.train['min_range'] = ""
        self.train['show_id'] = ""
        for i in self.ts_schedule[self.ts_schedule['show_counts'] != ""].index:
            show_id = self.ts_schedule.show_counts.iloc[i]
            time_slot = self.ts_schedule.방송일시.iloc[i]
            minrange = self.ts_schedule.min_range.iloc[i]
            idx = self.train[(self.train.방송일시 >= time_slot) & (
                        self.train.방송일시 < time_slot + datetime.timedelta(minutes=minrange))].index
            self.train.show_id.iloc[idx] = show_id
            self.train.min_range.iloc[idx] = minrange

    ############################
    ## primetime
    ############################
    def get_primetime(self, obj):
        """
        :**objective: get primetime for week and weekends respectively
        """
        self.train['primetime'] = 0
        prime_week = [9, 10, 11]
        prime_week2 = [16, 17, 18]
        prime_weekend = [7, 8, 9]
        prime_weekend2 = [13, 15, 16, 17]

        self.train.loc[(self.train['red'] == 0) & (self.train['holidays'] == 1) & (
            self.train['hours'].isin(prime_weekend)), 'primetime'] = 1
        self.train.loc[(self.train['red'] == 0) & (self.train['holidays'] == 1) & (
            self.train['hours'].isin(prime_weekend2)), 'primetime'] = 2

        self.train.loc[(self.train['holidays'] == 0) & (self.train['hours'].isin(prime_week)), 'primetime'] = 1
        self.train.loc[(self.train['holidays'] == 0) & (self.train['hours'].isin(prime_week2)), 'primetime'] = 2



    ############################
    ## Other characteristics
    ############################

    def check_pay(self):
        """
        :**objective: create 3 factor variable to identify payment methods ('ilsibul','muiza','none')
        """
        pay = []
        for i in range(0, len(self.train)):
            word = self.train['상품명'].iloc[i]
            if '(일)' in word or '일시불' in word:
                pay.append('ilsibul')
            elif '(무)' in word or '무이자' in word:
                pay.append('muiza')
            else:
                pay.append('none')
        self.train['pay'] = pay

    ############################
    ## External information
    ############################
    def add_categories(self):
        """
        :objective: add category columns
        :return: pandas dataframe
        """
        categories = pd.read_excel("../data/01/2019sales_added.xlsx")
        categories.상품코드 = categories.상품코드.dropna().astype(int).astype(str).str.zfill(6)
        categories.방송일시 = pd.to_datetime(categories.방송일시, format="%Y/%m/%d %H:%M")
        categories.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
        categories.rename(columns={' 취급액 ': '취급액'}, inplace=True)
        self.train = pd.merge(left=self.train,
                              right=categories[
                                  ['방송일시', '상품코드', 'brand', 'original_c', 'small_c', 'small_c_code', 'middle_c',
                                   'middle_c_code', 'big_c', 'big_c_code']],
                              how='inner', on=['방송일시', '상품코드'], sort=False)


    ############################
    ## Combine
    ############################

    def drop_na(self):
        """
        :objective: drop na rows and 취급액 == 50000
        """
        self.train = self.train[self.train['취급액'].notna()]
        self.train = self.train[self.train['취급액'] != 50000]

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


