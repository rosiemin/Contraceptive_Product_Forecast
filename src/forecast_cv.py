
import numpy as np
import pandas as pd
import calendar
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoCV,LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import univariate as uni
import clean_dta as clean
from tqdm import tqdm

def create_features(df, series_id, target_lst):
    """
    :param df: times series standard DataFrame, date feature, series id and target as columns
    :param series_id: string
    :param target: string
    :return:
    """
    
    lags = np.array([1, 2, 3])
    wins = np.array([2, 3, 4])
    for target in target_lst:
        for lag in lags:
            print(df.shape)
            df[f"{target}_lag_{lag}"] = df.groupby(series_id)[target].transform(lambda x: x.shift(lag))
        for win in wins:
            for lag in lags:
                print(win, lag, df.shape)
                df[f"{target}_rmean_{lag}_{win}"] = df.groupby(series_id)[f"{target}_lag_{lag}"].transform(
                    lambda x: x.rolling(win).mean())
        if target != 'stock_distributed':
            df.drop(target, inplace = True, axis = 1)
    print(df.shape)
    return df


def make_subset(df, n):
    df_lst= []
    for i in range(n):
        df_temp = df.copy()
        df_temp['forecast_dist'] = i+1
        df_temp['stock_distributed_new'] = df_temp['stock_distributed'].shift(-(i+1))
        df_lst.append(df_temp)
    return pd.concat(df_lst, axis = 0)



def CV(clf, X_train, y_train, X_test, y_test, seed = 42):
    dt_meta = DecisionTreeRegressor(max_depth=4,random_state = seed)
    if clf == 'Linear Regression':
        model = LinearRegression(normalize =True,fit_intercept =False)
    elif clf == 'Random Forest':
        model = RandomForestRegressor(random_state=seed,n_estimators = 100, verbose=seed, min_impurity_decrease = 0.15)
    elif clf == 'Light GBM':
        model = LGBMRegressor(objective ='regression',
                               #importance_type='weight',
                               boosting_type='rf',bagging_fraction=0.8,bagging_freq = 1,
                               n_leaves =31, n_estimators= 500, learning_rate =0.015,random_state=seed,metric='rmse',verbose=seed)
    elif clf == "Cat Boost":
        model = CatBoostRegressor(random_seed=seed,depth = 4)
    elif clf == 'Ada Boost':
        model =  AdaBoostRegressor(dt_meta,random_state = seed,n_estimators = 100)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {}
    metrics['RMSE'] = uni.RMSE(y_test, preds)
    metrics['MAE'] = uni.MAE(y_test, preds)
    metrics['MASE'] = uni.MASE(X_train, y_test, preds)

    return metrics, preds, model

def train_test_data(train, test, predictions = False):
    trainx3 = make_subset(train, 3)
    trainx3.dropna(inplace = True)
    X_trainx3 = trainx3.drop(['calendar', 'stock_distributed', 'ID'], axis = 1)
    fts = [c for c in X_trainx3.columns if c]
    X_trainx3 = X_trainx3.values
    y_trainx3 = trainx3['stock_distributed'].values
    if predictions:
        test['forecast_dist'] = np.where(test['month'] == 7, 1, np.where(test['month'] == 8, 2, 3))
        test.fillna(0, inplace = True)
        return X_trainx3, y_trainx3, test
    testx3 = make_subset(test, 3)

    testx3.dropna(inplace = True)

    X_testx3 = testx3.drop(['calendar', 'stock_distributed', 'ID'], axis = 1)
    X_testx3 = X_testx3.values
    y_testx3 = testx3['stock_distributed'].values

    return X_trainx3, y_trainx3, X_testx3, y_testx3, fts

if __name__ == "__main__":

    train, product, site = clean.load_data('../data/')

    df = clean.clean_data(train, product, site)
    df.sort_values(by=['site_code','product_code','calendar'], axis=0, inplace=True)
    
    # adding on the new data that we haven't seen for predictions
    ss = pd.read_csv('../data/SampleSubmission.csv')
    pred=pd.DataFrame(ss.ID.str.split('X',3).tolist(), columns = ['year','month','site_code','product_code'])

    for col in pred.columns:
        pred[col] = pred[col].str.strip()
    
    pred['month'] = pred['month'].astype(int)
    pred['year'] = pred['year'].astype(int)

    pred = clean.clean_data(pred, product, site)

    df_clean, bad_ids = clean.remove_short_df(df)
    # removing IDs that won't be predicted on based on my assumptions
    pred['new_id'] = pred['site_code'] + ' ' + pred['product_code']  
    pred = pred[~(pred['new_id'].isin(bad_ids))]
    pred.drop('new_id', inplace = True, axis = 1)

    df_new = pd.concat([df_clean, pred], axis = 0, ignore_index = True)

    df_new = df_new.drop(['region', 'district',
         'site_type', 'site_latitude', 'site_longitude',
        'product_type', 'site_product'], axis = 1)

    # # Removing the data that doesn't have any 2019 or is shorter than 6 months worth of data

    # # creating lag variables
    lag_lst = ['stock_adjustment', 'stock_end', 'average_monthly_consumption',
        'stock_ordered', 'stock_distributed']
    
    df_new1 = create_features(df_new, ['site_code', 'product_code'], lag_lst)

    le = LabelEncoder()

    # encoding labels w/ numbers to deal with strings
    for col in ['site_code','product_code']:
        df_new1[col]=  df_new1[col].astype('str')
        df_new1[col] = le.fit_transform(df_new1[col])
        


    # # need to segregate holdout set as to ensure no data leakage in the model
    prediction_set = df_new1[df_new1['calendar']>'2019-06-01']
    full_training = df_new1[~(df_new1['calendar']>'2019-06-01')]
    validation_set = df_new1[df_new1['calendar']>'2019-03-01']
    training = df_new1[~(df_new1['calendar']>'2019-03-01')]

    # # For cross validation I will be using the training set except for the most recent three months (01/01/2019 - 03/01/2019) - this will be the test set

    # running CV on training data
    train = training[training['year']!= 2019]
    test = training[training['year']==2019]

    X_train, y_train, X_test, y_test, fts = train_test_data(train, test)
    clf_lst = ['Linear Regression', 'Random Forest', 'Light GBM', "Cat Boost", 'Ada Boost']
    metrics_dict = {}
    pred_dict = {}
    for clf in clf_lst:
        m, p, modval = CV(clf, X_train, y_train, X_test, y_test)
        metrics_dict[clf] = m
        pred_dict[clf] = p
    

    # Let's test it on the validation set:
    X_train, y_train, X_test, y_test, fts = train_test_data(training, validation_set, False)
    metrics, preds, mod = CV('Random Forest', X_train, y_train, X_test, y_test, 42)

    # feature importances:

    fi = pd.Series(index = fts, data = mod.feature_importances_)
    _ = fi.sort_values(ascending = False)[:20][::-1].plot(kind = 'barh', figsize=(12, 8), fontsize=14)
    _ = plt.title('Feature Importance (Top 20)', fontsize = 14)

    # # Let's do our predictions:
    X_train, y_train, X_test = train_test_data(full_training, prediction_set, True)
    mod.fit(X_train, y_train)
    X_noID = X_test.drop(['ID', 'calendar'], axis = 1)
    predictions = mod.predict(X_noID.values)

    submission = X_test[['ID']].copy()
    submission['predictions'] = np.floor(predictions)
    submission.to_csv('../results/submission_predictions.csv')


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

    # time based validation - organize based on the date column.
    #decide on the validation period - 3 months
    # last three months is the true hold out
    # previous 3 months becomes validation
    #train data having the same duration of training data - if multiple validation windows
    # the newest data - 
