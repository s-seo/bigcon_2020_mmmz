import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math


## import NS - train data
train = pd.read_csv("../data/00/2019sales.csv", skiprows = 1)
#train.head()
#train.info()
#print("Are There Missing Data? :",train.isnull().any().any())
#print(train.isnull().sum())
#sum(train.iloc[:,7]==50000)
train.마더코드 = train.마더코드.dropna().astype(int).astype(str).str.zfill(6)
train.상품코드 = train.상품코드.dropna().astype(int).astype(str).str.zfill(6)
train.취급액 = train.취급액.dropna().astype(np.int64)
train.판매단가 = train.판매단가.dropna().astype(np.int64)



## 0. Assign 'volume' : sales/price
train = train.assign(volume = train.iloc[:,7]/train.iloc[:,6])
train.volume = train.volume.dropna().astype(np.int64)



## 1. min_start - starting minutes / factor 10 levels
min_start = []
dates = train.방송일시
for i in range(0,len(dates)):
    day = dates[i]
    a = day.split(' ')[1]
    b = a.split(':')[1]
    min_start.append(b)
list(set(min_start))
#len(min_start)



## 2. pay -  factor 3 levels
pay= []
names = train.상품명
for i in range(0,len(names)) :
    word = names[i]
    if '(일)' in word or '일시불' in word :
        pay.append('ilsibul')
    elif '(무)' in word or '무이자' in word :
        pay.append('muiza')
    else :
        pay.append('none')

#len(pay)
#pd.value_counts(pay)



## 3. holidays - weekends included / dummy
holiday = []
holiday_dates = ['2019.1.1', '2019.2.4','2019.2.5','2019.2.6',
                  '2019.3.1','2019.5.6','2019.6.6','2019.8.15',
                  '2019.9.12','2019.9.13','2019.10.3','2019.10.9',
                  '2019.12.25']
for i in range(0,len(dates)):
    day = dates[i]
    # whether date is 공휴
    a = day.split(' ')[0]
    # whether date is 주말
    year, month, day = (int(x) for x in a.split('.'))
    weekday = datetime.date(year, month, day)
    weekday = weekday.strftime("%A")
    if a in holiday_dates or weekday == 'Saturday' or weekday == 'Sunday':
        holiday.append(1)
    else: holiday.append(0)

#len(holiday)
#pd.value_counts(holiday)


## 4. days - factor
## 5. hours - factor 21 ㅣevels
hours = []
days = []

for i in range(0,len(dates)):
    day = dates[i]
    date = day.split(' ')[0]
    hour = day.split(' ')[1]

    # fill in days of week in 'days'
    year, month, day = (int(x) for x in date.split('.'))
    weekday = datetime.date(year, month, day)
    days.append(weekday.strftime('%A'))

    # fill in hours in 'hours'
    hours.append(int(hour.split(':')[0]))

#len(hours)
#len(days)
#pd.value_counts(hours)


## 6. hours_inweek - divide week into 168 hours / factor 168 levels
hours_inweek = []

for i in range(0,len(hours)):
    hour = hours[i]
    day = days[i]
    if day == 'Tuesday' :
        hours_inweek.append(hour+24)
    elif day == 'Wednesday' :
        hours_inweek.append(hour+24*2)
    elif day == 'Thursday' :
        hours_inweek.append(hour+24*3)
    elif day == 'Friday' :
        hours_inweek.append(hour+24*4)
    elif day == 'Saturday' :
        hours_inweek.append(hour+24*5)
    elif day == 'Sunday' :
        hours_inweek.append(hour+24*6)
    else :
        hours_inweek.append(hour)

#len(hours_inweek)



## 7. primetime - factor 4 levels
primetime = []
prime1 = [15,16,17,18]
prime2 = [19,20,21,22,23]
prime3 = [0,1,2,3,4,5,6]
prime4 = [7,8,9,10,11,12,13,14]

for i in range(0,len(hours)):
    hour = hours[i]
    if hour in prime1 :
        primetime.append('p1')
    elif hour in prime2 :
        primetime.append('p2')
    elif hour in prime3 :
        primetime.append('p3')
    else :
        primetime.append('p4')

#len(primetime)
#pd.value_counts(primetime)



## 8. luxury - dummy
luxury = [0]*len(hours)
train = train.assign(luxury = luxury)
train.loc[(train.brand=='명품') & (train.판매단가>=490000),'luxury'] = 1
train.loc[(train.original_c=='18k') & (train.판매단가>=490000),'luxury'] =1
#pd.value_counts(train.luxury)



## 9. on air - numeric/noise added
onair = pd.read_csv('../data/vrating_defined.csv',encoding = 'ISO-8859-1')
onair.columns = [ 'index','노출(분)','마더코드','상품코드','상품명','상품군','판매단가','취급액','DATE', 'TIME', 'rate_mean', 'rate_median',
       'rate_max', 'rate_sd' ]
for i in range(0,len(hours)):
    if math.isnan(onair.iloc[i,1]):
        onair.iloc[i,10] = onair.iloc[i-1,10]
    else:
        continue

# add noise to zero values
for i in range(0,len(hours)):
    val = onair.iloc[i,10]
    if val == 0 :
        onair.iloc[i,10] = np.random.uniform(0,1,1)[0]/1000000
rate_mean = onair.rate_mean
#len(rate_mean)
#plt.hist(onair.rate_mean, bins =50)



## Merge all
train_new = train.assign(min_start = min_start, pay = pay, holiday = holiday, hour = hours, weekday = days,
                        hours_inweek = hours_inweek, primetime = primetime, onair = rate_mean)


#train_new.head()
#on_air_hour = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1]
#on_air_hour = np.repeat(on_air_hour,60)
#len(on_air_hour)
#onair_sum = onair.assign(hour = on_air_hour)
