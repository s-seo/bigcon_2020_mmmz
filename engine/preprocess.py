import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# if separate df with all features is existent, set path
# otherwise merge raw df with Features module


def load_df(path=None):
    """
    :objective: load data
    :return: pandas dataframe
    """
    try:
        df = pd.read_pickle(path)
        return df.reset_index()
    except:
        print("check file directory")


def drop_useless(df):
    """
    :objective: drop useless features for model.
    :return: pandas dataframe
    """
    #useless features
    xcol = ['방송일시', '노출(분)', '마더코드', '상품명', 'exposed', 'ymd', 'volume',
            'years', 'days', 'hours', 'week_num', 'holidays', 'red', 'min_range', 'brand',
            'small_c_code', 'middle_c_code', 'big_c_code', 'sales_power']
    col = [x for x in df.columns if x in xcol]
    df = df.drop(columns=col)
    df = df.copy()
    return df


def check_na(df):
    """
    :objective: show na
    :return: columns with na / na counts
    """
    print(df.isnull().sum())


# 새로운 변수에 na 있으면 일단 imputation 따로 해야함. 아니면 여기 list에 변수 이름 추가해도 됨.
def na_to_zeroes(df):
    """
    :objective: Change all na's to zero.(just for original lag!)
    :return: pandas dataframe
    """
    xcol = [x for x in df.columns if
            x in lag_col1 + lag_col2 + ['mid_click_r', 'age30_middle', 'age40_middle', 'age50_middle',
                                        'age60above_middle', 'pc_middle', 'mobile_middle']]
    for col in xcol:
        df[col] = df[col].fillna(0)

    return df

## Encoding
# One-hot-Encoding
def run_onehot(df):
    """
    :objective: Perform ohe for categorical columns
    :return: pandas dataframe
    """
    cats_col = ['min_start','japp','parttime', 'primetime','exposed_t','상품군','weekdays', 'small_c','middle_c','big_c','pay','men']
    num = df.drop(columns = cats_col)
    X1 = df[cats_col]
    # Onehotencoder
    ohe = OneHotEncoder()
    ohe.fit(X1)
    onehotlabels = ohe.transform(X1).toarray()
    cat_labels = ohe.get_feature_names(['min_start','japp','parttime', 'primetime','exposed','상품군','weekdays', 'small_c','middle_c','big_c', 'pay','men'])
    cat = pd.DataFrame(onehotlabels, columns=cat_labels)
    df_ohe = num.join(cat)

    return df_ohe

# Label Encoding

def get_label_features(df):
    """
    :objective: Show features that need labelencoding
    :return: list
    """
    lab_col = df.select_dtypes(include=['object','category']).columns.tolist()

    return lab_col


def run_label_separately(df, colname):
    """
    :objective: Perform labelencoding for selected column(only one)
    :return: pandas dataframe with encoding only on selected column
    """
    if type(colname) == str:
        le = LabelEncoder()
        df[colname] = le.fit_transform(df[colname].values)
        return df
    else:
        print("Error! [colname] should be string type.")


def get_label_mapping(le):
    """
    :objective: Return a dict mapping labels to their integer values / Run right after 'run_label_separately' / le = a fitted SKlearn LabelEncoder
    :return: integer
    """
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

#integerMapping = get_label_mapping(le)
#integerMapping['Monday'] #하나만 보고 싶을 때
#이미 한번 인코딩을 한 변수에 또 labelencoding을 할 경우 mapping 못불러옴
#get_label_features 아웃풋 보고 한번씩 separate하게 인코딩 하고, 인코딩 한번 할 때마다 get_label_mapping(le)으로 확인하는거 추천
#get_label_features 확인 -> run_label_separately(df, colname) -> get_label_mapping(le) 저장 -> run_label_separately(df, colname) -> get_label_mapping(le) 저장


# mapping 안궁금하고 그냥 한꺼번에 다 돌리기
def run_label_all(df):
    """
    :objective: Perform labelencoding for all categorical/object columns
    :return: pandas dataframe
    """
    lab_col = df.select_dtypes(include=['object','category']).columns.tolist()
    for col in lab_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)

    return df


# remove outliers
def remove_outliers(df_train):
    """
    :objective: Remove outliers. Before dividing into X/y
    :return: pandas dataframe
    """
    numeric_colnum = df_train.columns.get_indexer(['판매단가', '취급액']).tolist()
    feature_set = df_train.iloc[:, numeric_colnum]
    # identify outliers in the training dataset
    iso = IsolationForest(n_estimators=50, max_samples=50, contamination=float(0.05), max_features=1.0)
    iso.fit(feature_set)
    pred = iso.predict(feature_set)
    feature_set['anomaly'] = pred
    outliers = feature_set.loc[feature_set['anomaly'] == -1]
    outlier_index = list(outliers.index)
    df_train = df_train.loc[~df_train.index.isin(outlier_index)].reset_index()

    return df_train

def run_preprocess(df):
    """
    :objective: Run Feature deletion, NA imputation, label encoding
    :return: pandas dataframe
    """
    df = drop_useless(df)
    df = na_to_zeroes(df)
    # df = remove_outliers(df)
    df = run_label_all(df)
    df1 = df.copy()
    return df1
