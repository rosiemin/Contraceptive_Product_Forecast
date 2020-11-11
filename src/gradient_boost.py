### Import necessary dependencies
import numpy as np
import pdb
import pandas as pd
import calendar
import datetime
from dateutil.relativedelta import relativedelta
import category_encoders as ce
from lightgbm import LGBMRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge,Lasso,LassoLarsCV
from  scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import univariate as uni
import clean_dta as clean
from tqdm import tqdm

train, product, site = clean.load_data('../data/')

df = clean.clean_data(train, product, site)
df.sort_values(by=['site_code','product_code','calendar'], axis=0, inplace=True)
# df.index = df.calendar


# df1, monthly = clean.group_and_subset_data(training, ['calendar'])

# columns to drop
# stock end b/c that' is hte same as stock initial T1
# stock order b/c that is part of stock received t1 and t2
# region and district since i'm looking by site
# average monthly consumption, unsure if this is across all products or past years aggregated by month, because of this i'm not quite sure what i should use
# remove data that has only so a few rows
# ARIMA is a rolling forecast - rolling/recursive - easy to build - but you magnify your error
# Forecase distance modeling - from this point, i'm going to make a prediction X months out all from this moment in time

# Duplicate dataset, for this row - what does next month look like separate row
#Forecast distnace 1 - tomorrow, 2, 2 days for now
# Triple dataset - 

df_trunc = df.drop(['region', 'district',
       'stock_adjustment', 'stock_end', 'average_monthly_consumption',
       'stock_ordered', 'site_type', 'site_latitude', 'site_longitude',
       'product_type', 'ID', 'site_product'], axis = 1)

# Removing the data that doesn't have any 2019 or is shorter than 6 months worth of data
df_new = clean.remove_short_df(df_trunc)


def create_features(df, series_id, target):
    """
    :param df: times series standard DataFrame, date feature, series id and target as columns
    :param series_id: string
    :param target: string
    :return:
    """
    lags = np.array([1, 2, 3])
    wins = np.array([2, 3, 4])
    for lag in lags:
        print(df.shape)
        df["lag_{}".format(lag)] = df.groupby(series_id)[target].transform(lambda x: x.shift(lag))
    for win in wins:
        for lag in lags:
            print(win, lag, df.shape)
            df["rmean_{}_{}".format(lag, win)] = df.groupby(series_id)["lag_{}".format(lag)].transform(
                lambda x: x.rolling(win).mean())
    print(df.shape)
    return df


# creating lag variables
df_new = create_features(df_new, ['site_code', 'product_code'], 'stock_distributed')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# encoding labels w/ numbers to deal with strings
for col in ['site_code','product_code']:
  df_new[col]=  df_new[col].astype('str')
  df_new[col]= le.fit_transform(df_new[col])




# need to segregate holdout set as to ensure no data leakage in the model
validation_set = df_new[df_new['calendar']>'2019-03-01']
training = df_new[~(df_new['calendar']>'2019-03-01')]

# For cross validation I will be using the training set except for the most recent three months (01/01/2019 - 03/01/2019) - this will be the test set

train = training[training['year']!= 2019]
test = training[training['year']==2019]

def make_subset(df, n):
    df_lst= []
    for i in range(n):
        df_temp = df.copy()
        df_temp['forecast_dist'] = i+1
        df_temp['stock_distributed_new'] = df_temp['stock_distributed'].shift(-(i+1))
        df_lst.append(df_temp)
    return pd.concat(df_lst, axis = 0)

trainx3 = make_subset(train, 3)
trainx3.dropna(inplace = True)
testx3 = make_subset(test, 3)


from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoCV,LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

seed = 42

# time based validation - organize based on the date column.
#decide on the validation period - 3 months
# last three months is the true hold out
# previous 3 months becomes validation
#train data having the same duration of training data - if multiple validation windows
# the newest data - 

def get_catboost():
  return CatBoostRegressor(random_seed=seed,depth = 4)


lin_reg = LinearRegression(normalize =True,fit_intercept =False)

svr = SVR(C = 1,kernel='poly', degree = 5) 

dt_meta = DecisionTreeRegressor(max_depth=4,random_state = seed)

rf = RandomForestRegressor(random_state=seed,n_estimators = 100, verbose=seed)

lgbm_regressor = LGBMRegressor(objective ='regression',
                               #importance_type='weight',
                               boosting_type='rf',bagging_fraction=0.8,bagging_freq = 1,
                               n_leaves =31, n_estimators= 500, learning_rate =0.015,random_state=seed,metric='rmse',verbose=seed)

cat_boost = get_catboost()

ada_boost = AdaBoostRegressor(dt_meta,random_state = seed,n_estimators = 100)

