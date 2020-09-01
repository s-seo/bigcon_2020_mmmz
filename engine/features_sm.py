## import libraries
# stat
import pandas as pd
import numpy as np

# data
import datetime

## load data
train = pd.read_csv("../data/00/2019sales.csv", skiprows = 1)
train.rename(columns={' 취급액 ': '취급액'}, inplace = True)
train['exposed']  = train['노출(분)']
# define data types
train.마더코드 = train.마더코드.astype(int).astype(str).str.zfill(6)
train.상품코드 = train.상품코드.astype(int).astype(str).str.zfill(6)
train.취급액 = train.취급액.str.replace(",","").astype(float)
train.판매단가 = train.판매단가.str.replace(",","").replace(' - ', np.nan).astype(float)
train.방송일시 = pd.to_datetime(train.방송일시, format="%Y/%m/%d %H:%M")
train.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace = True)
train['ymd'] = [d.date() for d in train["방송일시"]]

ts_schedule = train.copy().groupby('방송일시').first()
ts_schedule.reset_index(inplace = True)

def filter_jappingt(x):
    """
    :objective: round up 방송일시
    :param x: train row - pd.Dataframe
    :return: int
    """
    time = x['방송일시']
    if (time.minute < 30) & (time.hour == 0):
        rtn = time.hour
    elif time.minute >= 30:
        if time.hour == 23: rtn = 0
        else: rtn = time.hour + 1
    else:
        if time.hour == 0: rtn = 23
        else: rtn = time.hour
    return rtn

def fill_exposed_na():
    """
    :objective: fill out NA values on 'exposed' with mean(exposed)
    :return:pd.Dataframe with adjusted 'exposed' column
    """
    train["exposed"].fillna(train.groupby('방송일시')['exposed'].transform('mean'), inplace = True)

def get_ymd():
    """
    :objective: add 'ymd' variable to train dataset
    :return: pandas dataframe
    """
    t = 1
    while t < 9:
        for i in ts_schedule.ymd.unique():
            if i == datetime.date(2019,1,1): continue
            time_idx = ts_schedule[ts_schedule.ymd == i].index[0]
            first_show = ts_schedule.iloc[time_idx]
            last_show = ts_schedule.iloc[time_idx - 1]
            if (first_show['마더코드'] == last_show['마더코드']) & (first_show['방송일시'] <= last_show['방송일시'] + datetime.timedelta(minutes=last_show['exposed'])):
                ts_schedule.ymd.iloc[time_idx] = ts_schedule.ymd.iloc[time_idx - 1]

        t = t + 1

def timeslot():
    """
    :objective: get timeslot of each show
    :return: pandas dataframe
    """
    ts_schedule['parttime'] = ts_schedule.groupby(['ymd','상품코드']).cumcount()+1
    # 2019-10-27 06:00:00  manually update
    ts_schedule.parttime.iloc[19314:19317,] = (1,2,3)
    train['parttime'] = ""
    for i in range(0,len(ts_schedule)):
        timeslot = ts_schedule.방송일시.iloc[i]
        part = ts_schedule.parttime.iloc[i]
        train.loc[train.방송일시 == timeslot, 'parttime'] = part

def get_show_id():
    """
    :objective: get show id for each day
    :return: pandas dataframe
    """
    ts_schedule['show_counts'] = ""
    for i in ts_schedule.ymd.unique():
        rtn = ts_schedule[ts_schedule.ymd == i]
        slot_count = 0 #number of shows for each day
        for j in range(0,len(rtn)):
            if rtn['parttime'].iloc[j] ==  1:
                slot_count += 1
                idx = ts_schedule[ts_schedule.ymd == i].index[j]
                ts_schedule.show_counts.iloc[idx] = str(i) + " "+ str(slot_count)

def get_min_range():
    """
    :objective: get minutes aired for each show
    :return: pandas dataframe
    """
    ts_schedule['min_range'] = ""
    for i in range(0,len(ts_schedule)):
        if ts_schedule.parttime.iloc[i] == 1:
            min_dur = ts_schedule.exposed.iloc[i]
            j = i + 1
            if j == (len(ts_schedule)): break
            while ts_schedule.parttime.iloc[j] != 1:
                min_dur += ts_schedule.exposed.iloc[j]
                j += 1
                if j == (len(ts_schedule)): break
        ts_schedule.min_range.iloc[i:j] = min_dur

def add_showid_minran_to_train():
    """
    :objective: add show_id and min_range column to train data
    :return: pandas dataframe
    """
    train['min_range'] = ""
    train['show_id'] = ""
    for i in ts_schedule[ts_schedule['show_counts'] != ""].index:
        show_id = ts_schedule.show_counts.iloc[i]
        time_slot = ts_schedule.방송일시.iloc[i]
        minrange =  ts_schedule.min_range.iloc[i]
        idx = train[(train.방송일시 >= time_slot) & (train.방송일시 < time_slot + datetime.timedelta(minutes=minrange))].index
        train.show_id.iloc[idx] = show_id
        train.min_range.iloc[idx] = minrange

def get_sales_power():
    """
    :objective: get sales power of each product, sum(exposed time)/sum(sales volume)
    """
    train['sales_power'] = ""
    bp = train.groupby('상품코드').exposed.sum()/train.groupby('상품코드').volume.sum()
    for i in bp.index:
        train.sales_power.loc[train.상품코드 == i] = bp.loc[i]

def check_men_items():
    """
    :objective: create a dummy variable to identify products for men
    """
    train['men'] = 0
    train.men[train['상품명'].str.contains("남성")] = 1

def get_hour():
    """
    :objective: get hour
    """
    train['hours'] = train.방송일시.dt.hour

def freq_items():
    """
    :objective: identify frequently sold items by dummy variable "freq"
    """
    # define top ten frequently sold items list
    freq_list = train.groupby('상품코드').show_id.nunique().sort_values(ascending=False).index[1:10]
    train['freq'] = 0
    train.freq.loc[train.상품코드.isin(freq_list)] = 1

def drop_na():
    """
    :objective: drop na rows and 취급액 == 50000
    """
    rtn = train[train['취급액'].notna()]
    rtn = rtn[rtn['취급액']!= 50000]
    return rtn

train['japp'] = train.apply(filter_jappingt, axis=1)
train['volume'] = train['취급액']/train['판매단가']
fill_exposed_na()
get_ymd()
timeslot()
get_show_id()
get_min_range()
add_showid_minran_to_train()
get_sales_power()
check_men_items()
get_hour()
freq_items()
train = drop_na()

