# General imports
import warnings

warnings.filterwarnings("ignore")

# sklearn
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf

# visualize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from engine.utils import *
from engine.vars import *
#######################################################################
################################# Load Data ###########################
#######################################################################


## Import 3 types of dataset
## Descriptions:
#   - df_wd_lag : weekday / + lags
#   - df_wk_lag: weekend / + lags
#   - df_all_lag : all days / +lags

df_wd_lag = load_df(FEATURED_DATA_DIR + '/train_fin_wd_lag.pkl')
df_wk_lag = load_df(FEATURED_DATA_DIR + '/train_fin_wk_lag.pkl')
df_all_lag = load_df(FEATURED_DATA_DIR + '/train_fin_light_ver.pkl')

## Preprocessed datasets
df_wk_lag_PP = run_preprocess(df_wk_lag)
df_wd_lag_PP = run_preprocess(df_wd_lag)


#######################################################################
############################### Time  #################################
#######################################################################

colors = ['mistyrose', 'gold', 'yellow','darkorange','red', 'tomato', 'darksalmon', 'orange', 'orangered', 'indianred', 'lightcoral']

## 취급액 기준
df1 = df_all_lag.groupby(['japp','weekdays']).mean()['취급액'].unstack()
x = df1.index.tolist()
plt.figure(figsize= (15,10))
for col in df1.columns:
    if col not in ['Sunday', 'Saturday']:
        plt.plot(x, df1[col], marker = '', color = 'grey', linewidth = 2, alpha = 0.4)
    elif col == 'Sunday':
        plt.plot(x, df1[col], marker='', color='orange', linewidth=4, alpha=0.7)
    else:
        plt.plot(x, df1[col], marker='', color='tomato', linewidth=4, alpha=0.7)
pop_a = mpatches.Patch(color='orange', label='Saturday')
pop_b = mpatches.Patch(color='tomato', label='Sunday')
pop_c = mpatches.Patch(color='grey', label='Weekdays')
plt.legend(handles=[pop_a,pop_b,pop_c], fontsize = 27, loc =2)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.rcParams["axes.grid.axis"] ="y"
plt.rcParams["axes.grid"] = True
plt.savefig(PLOT_DIR + 'weekdaysSales.pdf')



## 방송횟수 기준
df2 = df_all_lag.groupby(['japp','weekdays']).apply(lambda grp : sum(grp.취급액)/len(np.unique(grp.show_id))).unstack()
x = df2.index.tolist()
plt.figure(figsize= (15,10))
for col in df2.columns:
    if col not in ['Sunday', 'Saturday']:
        plt.plot(x, df2[col], marker = '', color = 'grey', linewidth = 2, alpha = 0.4)
    elif col == 'Sunday':
        plt.plot(x, df2[col], marker='', color='orange', linewidth=4, alpha=0.7)
    else:
        plt.plot(x, df2[col], marker='', color='tomato', linewidth=4, alpha=0.7)
pop_a = mpatches.Patch(color='orange', label='Saturday')
pop_b = mpatches.Patch(color='tomato', label='Sunday')
pop_c = mpatches.Patch(color='grey', label='Weekdays')
plt.legend(handles=[pop_a,pop_b,pop_c], fontsize = 27, loc = 2)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.rcParams["axes.grid.axis"] ="y"
plt.rcParams["axes.grid"] = True
plt.savefig(PLOT_DIR + 'weekdaysSalespershow.pdf')



##  seasonality
df3 = df_all_lag.groupby(['hours_inweek','상품군']).mean()['취급액'].unstack()
x = df3.index.tolist()
plt.figure(figsize= (15,10))
plt.rcParams["axes.grid"] = False
legends = []
for col in df3.columns:
    i = int(df3.columns.get_indexer([col]))
    legends.append(mpatches.Patch(color=colors[i], label=''))
    plt.plot(x, df3[col], marker='', color=colors[i], linewidth=2, alpha=0.7)
plt.legend(handles=legends, fontsize = 27, loc = 2)
xcoords = [24, 48, 72, 96, 120, 144, 168]
for xc in xcoords:
    plt.axvline(x=xc, color = 'silver', ls = '-.', linewidth = 2)
df31 = df_all_lag.groupby(['hours_inweek']).mean()['취급액'].reset_index()
plt.plot(df31['hours_inweek'], df31['취급액'], marker='', color='dimgrey', linewidth=4, alpha=0.7)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Hours in Week', fontsize=25)
plt.ylabel('Sales', fontsize=25)
plt.xlim(0,168)
plt.ylim((0,1.5e+08))
plt.savefig(PLOT_DIR + 'seasonal.pdf')

## seasonality in details
groups_ori = df_all_lag.groupby(['months','days','show_id','original_c'])['취급액','volume'].mean().reset_index()


def seasonal(item, rank):
    """
    :objective: get original c's sales
    :param item: str - level of origina_c
    :param rank: int - take first n sales
    """
    a = groups_ori[(groups_ori.original_c == item)]
    a_sort = a.sort_values(by='취급액', axis=0, ascending=False)
    result = pd.DataFrame(a_sort.head(rank))
    return result


def seasonal_plot(item):
    """
    :objective: draw monthly sales plot for each item
    :param item: str - level of origina_c
    """
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    a = groups_ori[(groups_ori.original_c == item)]
    fig, ax = plt.subplots()
    ax.hist(a.months, bins=12)
    ax.set_xticks(x)
    plt.title(item)
    plt.show()

#
# for i in range(len(df_all_lag.original_c.unique())):
#     print(seasonal(df_all_lag.original_c.unique()[i], 30))
#     seasonal_plot(df_all_lag.original_c.unique()[i])

#######################################################################
############################# 기존편성 관련  #############################
#######################################################################

## 취급액 값
df4 = df_all_lag.groupby(['japp','상품군']).sum()['취급액'].unstack().fillna(0)
x = df4.index.tolist()
plt.rcParams["figure.figsize"] = (14,14)
p = df4.plot(kind='bar',stacked=True, legend = None).get_figure()
plt.xticks(fontsize = 28, rotation = 0)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Sales', fontsize=25)
p.savefig(PLOT_DIR + 'HourlySales.pdf')

## 취급액 비율
snum = df4.sum(axis =1).reset_index()
x = df4.index.tolist()
for i in range(0,21):
    df4.iloc[i,:] = df4.iloc[i,:]/snum.iloc[i,1]
plt.rcParams["figure.figsize"] = (14,14)
p = df4.plot(kind='bar',stacked=True,legend = None).get_figure()
plt.xticks(fontsize = 28, rotation = 0)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Sales', fontsize=25)
p.savefig(PLOT_DIR + 'HourlySalesProp.pdf')


## 방송횟수 값
df5 = df_all_lag.groupby(['japp','상품군']).count()['취급액'].unstack().fillna(0)
x = df5.index.tolist()
plt.rcParams["figure.figsize"] = (14,14)
p = df5.plot(kind='bar',stacked=True, legend = None).get_figure()
plt.xticks(fontsize = 28, rotation = 0)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Count', fontsize=25)
plt.savefig(PLOT_DIR + 'HourlyCount.pdf')

## 방송횟수 비율
snum = df5.sum(axis =1).reset_index()
x = df5.index.tolist()
for i in range(0,21):
    df5.iloc[i,:] = df5.iloc[i,:]/snum.iloc[i,1]
plt.rcParams["figure.figsize"] = (14,14)
p = df5.plot(kind='bar',stacked=True, legend = None).get_figure()
plt.xticks(fontsize = 28, rotation = 0)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Count', fontsize=25)
p.savefig(PLOT_DIR + 'HourlyCountProp.pdf')


## 판매량 값
df6 = df_all_lag.groupby(['japp','상품군']).sum()['volume'].unstack().fillna(0)
x = df6.index.tolist()
plt.rcParams["figure.figsize"] = (14,14)
p = df6.plot(kind='bar',stacked=True, legend = None).get_figure()
plt.xticks(fontsize = 28, rotation = 0)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Sales', fontsize=25)
p.savefig(PLOT_DIR + 'HourlyVolume.pdf')


## 판매량 비율
snum = df6.sum(axis =1).reset_index()
x = df6.index.tolist()
for i in range(0,21):
    df6.iloc[i,:] = df6.iloc[i,:]/snum.iloc[i,1]
plt.rcParams["figure.figsize"] = (14,14)
p = df6.plot(kind='bar',stacked=True, legend = None).get_figure()
plt.xticks(fontsize = 28, rotation = 0)
plt.yticks(fontsize = 20)
plt.xlabel('Hours', fontsize=25)
plt.ylabel('Count', fontsize=25)
p.savefig(PLOT_DIR + 'HourlyVolumeProp.pdf')



## train
grp = df_all_lag.groupby('마더코드').count()['방송일시'].reset_index().sort_values('방송일시', ascending = False).iloc[:10,:]
names = []
sales = []
mean_sales = []
for i in range(0,10):
    mcode = grp['마더코드'].iloc[i]
    names.append(df_all_lag[df_all_lag['마더코드']==mcode].상품명.iloc[0])
    sales.append(sum(df_all_lag[df_all_lag['마더코드']==mcode].취급액))
    mean_sales.append(np.mean(df_all_lag[df_all_lag['마더코드']==mcode].취급액))
grp['names'] = names
grp['cum_sales'] = sales
grp['mean_sales'] = mean_sales
grp

# 방송횟수 기준
sum(grp.방송일시)/35379
10/df_all_lag.마더코드.nunique()

# 취급액 기준
sum(grp.cum_sales)/sum(df_all_lag.취급액)


## test
test_eda = pd.read_pickle(LOCAL_DIR + '/data/20/test_fin_light_ver.pkl')
grp = test_eda.groupby('마더코드').count()['방송일시'].reset_index().sort_values('방송일시', ascending = False).iloc[:10,:]
#grp = test_eda.groupby('마더코드').show_id.nunique().reset_index().sort_values('show_id',ascending = False).iloc[:10,:]
names = []
for i in range(0,10):
    mcode = grp['마더코드'].iloc[i]
    names.append(test_eda[test_eda['마더코드']==mcode].상품명.iloc[0])
grp['names'] = names
grp


## Agriculture
grp1 = df_all_lag.groupby('마더코드').mean()['취급액'].reset_index().sort_values('취급액', ascending = False)
grp2 = df_all_lag[df_all_lag.상품군 == '농수축'].groupby('마더코드').mean()['취급액'].reset_index().sort_values('취급액', ascending = False)
plt.figure(figsize= (15,10))
sns.distplot(grp1['취급액'], color = 'grey')
sns.distplot(grp2['취급액'], color = 'orange')

legends = []
legends.append(mpatches.Patch(color='grey', label='All'))
legends.append(mpatches.Patch(color='orange', label='Agriculture'))
plt.legend(handles=legends, fontsize = 27)
plt.xlabel('Mean Sales', fontsize=15)
plt.savefig(PLOT_DIR + 'salesdist.pdf')


plt.figure(figsize= (15,10))
sns.distplot(df_all_lag.판매단가, color = 'grey', bins = 1000)
sns.distplot(df_all_lag[df_all_lag.상품군 == '농수축'].판매단가, color = 'orange',bins = 10)
legends = []
legends.append(mpatches.Patch(color='grey', label='All'))
legends.append(mpatches.Patch(color='orange', label='Agriculture'))
plt.legend(handles=legends, fontsize = 27)
plt.xlim(0,1000000)
plt.ylim((0,4e-5))
plt.xlabel('Mean Price', fontsize=15)
plt.savefig(PLOT_DIR + 'pricedist.pdf')




###############################################################################
## 3. Train vs test

sum = 0
for name in test_eda.상품코드.unique().tolist():
    if name in df_all_lag.상품코드.unique().tolist():
        sum += 1
    else: continue
sum


###############################################################################
## 4. vratings
v_eda = pd.read_excel(LOCAL_DIR + '/data/00/2019vrating.xlsx',skiprows = 1)
v_eda = v_eda.drop(columns = ['시간대'])
v_1 = v_eda.iloc[:,5]
plt.figure(figsize= (20,5))
plt.plot(range(0,1441), v_1, marker='', color='dimgrey', linewidth=2, alpha=0.7)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Vratings', fontsize=25)
plt.savefig(PLOT_DIR + 'vratings.pdf')



###############################################################################
## 5. Feature Engineering

## weather
plt.figure(figsize= (20,5))
plt.plot(range(0,35379), df_all_lag.temp_diff_s, marker='', color='dimgrey', linewidth=2, alpha=0.7)
plt.xlabel('Time', fontsize=25)
plt.ylabel('temp_diff', fontsize=25)
plt.savefig(PLOT_DIR + 'tempdiff.pdf')



## lag
df_time  = df_all_lag.groupby('ymd').mean()['취급액'].reset_index()
x = range(0,366)

reg = LinearRegression()
reg.fit(pd.Series(x).values.reshape(-1,1), df_time['취급액'])
fitted = reg.coef_*x+reg.intercept_
resid = df_time['취급액']-fitted
plt.figure(figsize= (5,5))
plt.scatter(x,resid,c='orange')
plt.axhline(y=0, color = 'silver', ls = '-', linewidth = 2)
plt.savefig(PLOT_DIR + 'resid.pdf')

plt.figure(figsize= (4,3))
plot_acf(resid)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Residuals', fontsize=15)
plt.savefig(PLOT_DIR + 'acf.pdf')

plt.figure(figsize= (6,4))
plt.plot(x,df_time['취급액'],color = 'grey')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Sales', fontsize=15)
plt.savefig(PLOT_DIR + 'plotts.pdf')

