## import libraries
# stat
import pandas as pd
import numpy as np
import math
import random
import statistics as st

# data
import datetime

class features_p1:
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

    ##################################
    ## onair time/order info variables
    ##################################

    def get_time(self):
        """
        :** objective: get year, month, day, hours
        """
        self.train['years'] = self.train.방송일시.dt.year
        self.train['months'] = self.train.방송일시.dt.month
        self.train['days'] = self.train.방송일시.dt.day
        self.train['hours'] = self.train.방송일시.dt.hour

    def get_weekday(self):
        """
        :** objective: get weekday
        """
        self.train['weekdays'] = self.train.방송일시.dt.day_name()

    def get_hours_inweek(self):
        """
        :** objective: get hours by week (1~168)
        """
        hours_inweek = []
        for i in range(0, len(self.train)):
            hr = self.train['hours'].iloc[i]
            dy = self.train['weekdays'].iloc[i]
            if dy == 'Tuesday' :
                hours_inweek.append(hr+24)
            elif dy == 'Wednesday' :
                hours_inweek.append(hr+24*2)
            elif dy == 'Thursday' :
                hours_inweek.append(hr+24*3)
            elif dy == 'Friday' :
                hours_inweek.append(hr+24*4)
            elif dy == 'Saturday' :
                hours_inweek.append(hr+24*5)
            elif dy == 'Sunday' :
                hours_inweek.append(hr+24*6)
            else :
                hours_inweek.append(hr)
        self.train['hours_inweek'] = hours_inweek

    def get_holidays(self):
        """
        :** objective: create a dummy variable for holidays (weekends + red)
        """
        holidays = []
        holiday_dates = ['2019-01-01', '2019-02-04','2019-02-05','2019-02-06',
                          '2019-03-01','2019-05-06','2019-06-06','2019-08-15',
                          '2019-09-12','2019-09-13','2019-10-03','2019-10-09',
                          '2019-12-25']
        for i in range(0, len(self.train)):
            dt = str(self.train['ymd'].iloc[i])
            dy = self.train['weekdays'].iloc[i]
        if dt in holiday_dates or dy == 'Saturday' or dy == 'Sunday':
            holidays.append(1)
        else: holidays.append(0)
        self.train['holidays'] = holidays

    def get_red_days(self):
        """
        :** objective: create a dummy variable for just red
        """
        red = []
        holiday_dates = ['2019-01-01', '2019-02-04','2019-02-05','2019-02-06',
                          '2019-03-01','2019-05-06','2019-06-06','2019-08-15',
                          '2019-09-12','2019-09-13','2019-10-03','2019-10-09',
                          '2019-12-25']
        for i in range(0, len(self.train)):
            dt = str(self.train['ymd'].iloc[i])
        if dt in holiday_dates:
            red.append(1)
        else: red.append(0)
        self.train['red'] = red

    def get_weekends(self):
        """
        :** objective: create a dummy variable for just weekends
        """
        self.train['weekends'] = 0
        self.train.loc[(self.train['red']==0) & (self.train['holidays']==1),'weekends'] =1


    def get_min_start(self):
        """
        :** objective: get startig time (min)
        """
        self.train['min_start'] = self.train.방송일시.dt.minute
        #list(set(train.방송일시.dt.minute)) #unique

    def filter_jappingt(self):
        """
        :objective: round up 방송일시
        """
        japp = []
        for i in range(0, len(self.train)):
            time = self.train['방송일시'].iloc[i]
            if (time.minute < 30) & (time.hour == 0):
                rtn = time.hour
            elif time.minute >= 30:
                if time.hour == 23: rtn = 0
                else: rtn = time.hour + 1
            else:
                if time.hour == 0: rtn = 23
                else: rtn = time.hour
            japp.append(rtn)
        self.train['japp'] = japp

    def fill_exposed_na(self):
        """
        :objective: fill out NA values on 'exposed' with mean(exposed)
        :return:pd.Dataframe with adjusted 'exposed' column
        """
        self.train["exposed"].fillna(self.train.groupby('방송일시')['exposed'].transform('mean'), inplace = True)

    def get_ymd(self):
        """
        :objective: add 'ymd' variable to train dataset
        :return: pandas dataframe
        """
        t = 1
        while t < 9:
            for i in self.ts_schedule.ymd.unique():
                if i == datetime.date(2019,1,1): continue
                time_idx = self.ts_schedule[self.ts_schedule.ymd == i].index[0]
                first_show = self.ts_schedule.iloc[time_idx]
                last_show = self.ts_schedule.iloc[time_idx - 1]
                if (first_show['마더코드'] == last_show['마더코드']) & (first_show['방송일시'] <= last_show['방송일시'] + datetime.timedelta(minutes=last_show['exposed'])):
                    self.ts_schedule.ymd.iloc[time_idx] = self.ts_schedule.ymd.iloc[time_idx - 1]

            t = t + 1

    def timeslot(self):
        """
        :objective: get timeslot of each show
        :return: pandas dataframe
        """
        self.ts_schedule['parttime'] = self.ts_schedule.groupby(['ymd','상품코드']).cumcount()+1
        # 2019-10-27 06:00:00  manually update
        self.ts_schedule.parttime.iloc[19314:19317,] = (1,2,3)
        self.train['parttime'] = ""
        for i in range(0,len(self.ts_schedule)):
            timeslot = self.ts_schedule.방송일시.iloc[i]
            part = self.ts_schedule.parttime.iloc[i]
            self.train.loc[self.train.방송일시 == timeslot, 'parttime'] = part

    def get_show_id(self):
        """
        :objective: get show id for each day
        :return: pandas dataframe
        """
        self.ts_schedule['show_counts'] = ""
        for i in self.ts_schedule.ymd.unique():
            rtn = self.ts_schedule[self.ts_schedule.ymd == i]
            slot_count = 0 #number of shows for each day
            for j in range(0,len(rtn)):
                if rtn['parttime'].iloc[j] ==  1:
                    slot_count += 1
                    idx = self.ts_schedule[self.ts_schedule.ymd == i].index[j]
                    self.ts_schedule.show_counts.iloc[idx] = str(i) + " "+ str(slot_count)

    def get_min_range(self):
        """
        :objective: get minutes aired for each show
        :return: pandas dataframe
        """
        self.ts_schedule['min_range'] = ""
        for i in range(0,len(self.ts_schedule)):
            if self.ts_schedule.parttime.iloc[i] == 1:
                min_dur = self.ts_schedule.exposed.iloc[i]
                j = i + 1
                if j == (len(self.ts_schedule)): break
                while self.ts_schedule.parttime.iloc[j] != 1:
                    min_dur += self.ts_schedule.exposed.iloc[j]
                    j += 1
                    if j == (len(self.ts_schedule)): break
            self.ts_schedule.min_range.iloc[i:j] = min_dur

    def add_showid_minran_to_train(self):
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
            idx = self.train[(self.train.방송일시 >= time_slot) & (self.train.방송일시 < time_slot + datetime.timedelta(minutes=minrange))].index
            self.train.show_id.iloc[idx] = show_id
            self.train.min_range.iloc[idx] = minrange


    ############################
    ## primetime
    ############################
    def get_primetime(self):
        """
        :**objective: get primetime for week and weekends respectively
        """
        self.train['primetime'] = 0
        prime_week = [9,10,11]
        prime_week2 = [16,17,18]
        prime_weekend = [7,8,9]
        prime_weekend2 = [13,15,16,17]

        self.train.loc[(self.train['red']==0) & (self.train['holidays']==1) & (self.train['hours'].isin(prime_weekend)),'primetime'] =1
        self.train.loc[(self.train['red']==0) & (self.train['holidays']==1) & (self.train['hours'].isin(prime_weekend2)),'primetime'] = 2

        self.train.loc[(self.train['holiday']==0) & (self.train['hours'].isin(prime_week)),'primetime'] = 1
        self.train.loc[(self.train['holiday']==0) & (self.train['hours'].isin(prime_week2)),'primetime'] = 2


    ############################
    ## sales/volume power variables
    ############################
    def get_sales_power(self):
        """
        :objective: get sales power of each product, sum(exposed time)/sum(sales volume)
        """
        self.train['sales_power'] = ""
        bp = self.train.groupby('상품코드').exposed.sum()/self.train.groupby('상품코드').volume.sum()
        for i in bp.index:
            self.train.sales_power.loc[self.train.상품코드 == i] = bp.loc[i]

    def freq_items(self):
        """
        :objective: identify frequently sold items by dummy variable "freq"
        """
        # define top ten frequently sold items list
        freq_list = self.train.groupby('상품코드').show_id.nunique().sort_values(ascending=False).index[1:10]
        self.train['freq'] = 0
        self.train.freq.loc[self.train.상품코드.isin(freq_list)] = 1


    ############################
    ## Other characteristics
    ############################
    def check_men_items(self):
        """
        :objective: create a dummy variable to identify products for men
        """
        self.train['men'] = 0
        self.train.men[self.train['상품명'].str.contains("남성")] = 1

    def check_luxury_items(self):
        """
        :**objective: create a dummy variable to identify products with selling price >= 490,000
        """
        self.train['luxury'] = 0
        self.train.loc[self.train['판매단가']>=490000, 'luxury'] = 1

    def check_pay(self):
        """
        :**objective: create 3 factor variable to identify payment methods ('ilsibul','muiza','none')
        """
        pay = []
        for i in range(0,len(self.train)) :
            word = self.train['상품명'].iloc[i]
            if '(일)' in word or '일시불' in word :
                pay.append('ilsibul')
            elif '(무)' in word or '무이자' in word :
                pay.append('muiza')
            else :
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
        output = pd.merge(left=self.train,
                          right=categories[['방송일시', '상품코드', 'brand', 'original_c', 'small_c', 'middle_c', 'big_c']],
                          how='left', on=['방송일시', '상품코드'], sort=False)
        return output

    def add_vratings(self):
        """
        :**objective: add vratings by rate mean
        """
        onair = pd.read_csv("../data/vrating_defined.csv")
        onair.상품코드 = onair.상품코드.dropna().astype(int).astype(str).str.zfill(6)
        onair['방송일시'] = onair[['DATE','TIME']].agg(' '.join, axis=1)
        onair['방송일시'] = pd.to_datetime(onair.방송일시, format="%Y/%m/%d %H:%M")
        onair.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)

        #impute rate mean nan
        random.seed(100)
        for i in range(0,len(self.train)):
            if math.isnan(onair.iloc[i,1]):
                onair['rate_mean'].iloc[i] = onair['rate_mean'].iloc[i-1]
            else:
                continue

        # add noise to zero values
        for i in range(0,len(self.train)):
            val = onair['rate_mean'].iloc[i]
            if val == 0 :
                onair['rate_mean'].iloc[i] = np.random.uniform(0,1,1)[0]/1000000
        rate_mean = onair['rate_mean']
        self.train['vratings'] = rate_mean

    def drop_na(self):
        """
        :objective: drop na rows and 취급액 == 50000
        """
        rtn = self.train[self.train['취급액'].notna()]
        rtn = rtn[rtn['취급액']!= 50000]
        return rtn

    def run_all(self):
        self.get_time()
        self.get_weekday()
        self.get_hours_inweek()
        self.get_holidays()
        self.get_red_days()
        self.get_weekends()
        self.get_min_start()
        self.filter_jappingt()
        self.fill_exposed_na()
        self.get_ymd()
        self.timeslot()
        self.get_show_id()
        self.get_min_range()
        self.add_showid_minran_to_train()
        self.get_primetime()
        self.get_sales_power()
        self.freq_items()
        self.check_men_items()
        self.check_luxury_items()
        self.check_pay()
        self.add_vratings()
        self.train = self.add_categories()
        self.train = self.drop_na()
        return self.train



t = features_p1()
train = t.run_all()
