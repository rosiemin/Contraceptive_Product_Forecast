
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tqdm import tqdm
import numpy as np
import pandas as pd
import calendar
import datetime
import itertools
import clean_dta as clean
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
from fbprophet import Prophet
import univariate as uni
from statsmodels.tsa.ar_model import AutoReg
from random import random
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot') # I also like fivethirtyeight'
matplotlib.rcParams.update({'font.size': 16, 'font.family': 'sans'})

def validation_run_all_products_sites(all_names, df_use):
    product_lst = []
    metrics_lst = []
    train_lst= []
    for product in tqdm(all_names):
        sub_df = df_use[(df_use['product_code']== product.split()[1]) & (df_use['site_code']== product.split()[0])]
        print('train test split)')
        # creating rows for all dates, to fill in gaps to reduce errors in models
        r = pd.date_range(start=df_use.calendar.min(), end=df_use.calendar.max(), freq = 'MS')
        sub_df = sub_df.reindex(r).fillna(0).rename_axis('calendar')
        sub_df['site_code'] = product.split()[0]
        sub_df['product_code'] = product.split()[1]
        sub_df['calendar'] = sub_df.index
        # Had to add one because some models won't work with 0's
        sub_df['stock_distributed'] = sub_df['stock_distributed'].copy() + 1

        train, test = uni.train_test_split(sub_df, 0.8)

        # uni.test_stationarity(sub_df.stock_distributed.dropna(), sub_df.calendar)
        print('identifying p, d, q')
        try:
            params, params_seasonal = uni.p_d_q(sub_df['stock_distributed'])
        except:
            params = (0, 1, 1)
            params_seasonal = (0, 1, 1, 12)  

        print('trying SARIMA')
        test = uni.SARIMA(sub_df, test['calendar'].min(), params, params_seasonal, test, 'SARIMA')
        test['SARIMA'] = test['SARIMA'].copy() - 1

        if sum(sub_df['stock_distributed']) == len(sub_df):
            print('Trying Holt Winders')
            test['HW-3'] = 0
        else:
            print('Trying Holt Winders')
            test = uni.holt_winters(sub_df, 3, test, 'HW-3')
            test['HW-3'] = test['HW-3'].copy() - 1
        if len(sub_df)>3:
            print('Trying FB Prophet')
            test = uni.fb_prophet(train, test, len(test))
            test['fbProphet'] = test['fbProphet'].copy() - 1
        else:
            print('Trying FB Prophet')
            test['fbProphet'] = 0
        
        # removing the 1 from the original dataset to keep consistency
        test['stock_distributed'] = test['stock_distributed'].copy() - 1
        test.fillna(0, inplace = True)

        print('Calculating metrics')
        metrics = uni.cross_val(test, train, ['SARIMA', 'HW-3', 'fbProphet'], [f'{product}_SARIMA', f'{product}_Holt-Winters 3', f'{product}_FB Prophet'])

        print('identifying best model for this data and saving data')
        product_lst.append(test)
        metrics_lst.append(metrics[metrics['MASE'] == metrics["MASE"].min()])
        train_lst.append(train)

    all_train = pd.concat(train_lst, axis = 0).to_csv('all_train.csv')
    all_test = pd.concat(product_lst, axis = 0).to_csv('all_test.csv')
    all_metrics = pd.concat(metrics_lst, axis = 0).to_csv('all_metrics.csv')

    return all_train, all_test, all_metrics

def predict_SARIMA(train, order, seasonal_order, s, p):
    fit1 = sm.tsa.statespace.SARIMAX(train.stock_distributed, order=order, seasonal_order=seasonal_order,enforce_stationarity=False,
        enforce_invertibility=False, disp = 0).fit(use_boxcox=True)
    try:
        results = fit1.predict(start = train.index.max(), end=pd.to_datetime('2019-09-01'), dynamic=True)
    except IndexError: 
        pass

    res = results.reset_index()
    res.columns = ['date', 'prediction']
    res.date = pd.to_datetime(res.date)
    res['month'] = res.date.dt.month
    res['year'] = res.date.dt.year
    res['product'] = p 
    res['site'] = s
    return res 

def get_all_predictions(df):
    ss = pd.read_csv('data/SampleSubmission.csv')
    ss['year'] = ss.ID.str.split(' X ').str[0]
    ss['month'] = ss.ID.str.split(' X ').str[1]
    ss['site'] = ss.ID.str.split(' X ').str[2]
    ss['product'] = ss.ID.str.split(' X ').str[3]

    all_preds = []

    for s, p in tqdm(zip(ss['site'], ss['product'])):
        sub_df = df_use[(df_use['product_code']== p) & (df_use['site_code']== s)]

        print('identifying p, d, q')
        try:
            params, params_seasonal = uni.p_d_q(sub_df['stock_distributed'])
        except:
            params = (0, 1, 1)
            params_seasonal = (0, 1, 1, 12)  
        
        all_preds.append(predict_SARIMA(sub_df, params, params_seasonal, s, p))
    
    final_df = pd.concat(all_preds, axis = 0)

    return final_df



if __name__ == "__main__":
    

    train, product, site = clean.load_data('../data/')

    df = clean.clean_data(train, product, site)
    df = clean.remove_short_df(df)

    df_use = df[['calendar', 'site_code', 'product_code', 'stock_distributed', 'site_product']].copy()

    df_use.sort_values(['site_code','product_code', 'calendar'], inplace = True)

    df_use = df_use.set_index('calendar',drop = False)

    product_names = df_use.product_code.unique()
    site_names = df_use.site_code.unique()

    all_names = df_use['site_product'].unique().tolist()
    df_use.drop('site_product', inplace = True, axis = 1)

    product_lst = []
    metrics_lst = []
    train_lst= []
    for product in tqdm(all_names):
        sub_df = df_use[(df_use['product_code']== product.split(' X ')[1]) & (df_use['site_code']== product.split(' X ')[0])]
        print('train test split)')
        # creating rows for all dates, to fill in gaps to reduce errors in models
        r = pd.date_range(start=df_use.calendar.min(), end=df_use.calendar.max(), freq = 'MS')
        sub_df = sub_df.reindex(r).fillna(0).rename_axis('calendar')
        sub_df['site_code'] = product.split(' X ')[0]
        sub_df['product_code'] = product.split(' X ')[1]
        sub_df['calendar'] = sub_df.index
        # Had to add one because some models won't work with 0's
        sub_df['stock_distributed'] = sub_df['stock_distributed'].copy() + 1

        train, test = uni.train_test_split(sub_df, 0.8)

        # uni.test_stationarity(sub_df.stock_distributed.dropna(), sub_df.calendar)
        print('identifying p, d, q')
        try:
            params, params_seasonal = uni.p_d_q(sub_df['stock_distributed'])
        except:
            params = (0, 1, 1)
            params_seasonal = (0, 1, 1, 12)  

        print('trying SARIMA')
        test = uni.SARIMA(sub_df, test['calendar'].min(), params, params_seasonal, test, 'SARIMA')
        test['SARIMA'] = test['SARIMA'].copy() - 1

        if sum(sub_df['stock_distributed']) == len(sub_df):
            print('Trying Holt Winders')
            test['HW-3'] = 0
        else:
            print('Trying Holt Winders')
            test = uni.holt_winters(sub_df, 14, test, 'HW-3')
            test['HW-3'] = test['HW-3'].copy() - 1
        if len(sub_df)>3:
            print('Trying FB Prophet')
            test = uni.fb_prophet(train, test, len(test))
            test['fbProphet'] = test['fbProphet'].copy() - 1
        else:
            print('Trying FB Prophet')
            test['fbProphet'] = 0
        
        # removing the 1 from the original dataset to keep consistency
        test['stock_distributed'] = test['stock_distributed'].copy() - 1
        test.fillna(0, inplace = True)

        print('Calculating metrics')
        metrics = uni.cross_val(test, train, ['SARIMA', 'HW-3', 'fbProphet'], [f'{product}_SARIMA', f'{product}_Holt-Winters 3', f'{product}_FB Prophet'])

        print('identifying best model for this data and saving data')
        product_lst.append(test)
        metrics_lst.append(metrics[metrics['MASE'] == metrics["MASE"].min()])
        train_lst.append(train)

    all_train = pd.concat(train_lst, axis = 0)
    all_test = pd.concat(product_lst, axis = 0)
    all_metrics = pd.concat(metrics_lst, axis = 0)



    # all_train, all_test, all_metrics = validation_run_all_products_sites(all_names, df_use)

    # running metrics across all tests for each model
    total_metrics = uni.cross_val(all_test, all_train, ['SARIMA', 'HW-3', 'fbProphet'], ['SARIMA', 'HW', 'FB'])

    # final = get_all_predictions(df_use)


    # ss = pd.read_csv('../data/SampleSubmission.csv')
    # ss['year'] = ss.ID.str.split(' X ').str[0]
    # ss['month'] = ss.ID.str.split(' X ').str[1]
    # ss['site'] = ss.ID.str.split(' X ').str[2]
    # ss['product'] = ss.ID.str.split(' X ').str[3]

    # all_preds = []

    # for p in tqdm(all_names):
    #     sub_df = df_use[(df_use['product_code']== p.split(' X ')[1]) & (df_use['site_code']== p.split(' X ')[0])]
        
    #     r = pd.date_range(start=df_use.calendar.min(), end=df_use.calendar.max(), freq = 'MS')
    #     sub_df = sub_df.reindex(r).fillna(0).rename_axis('calendar')
    #     sub_df['site_code'] = p.split(' X ')[0]
    #     sub_df['product_code'] = p.split(' X ')[1]
    #     sub_df['calendar'] = sub_df.index
    #     sub_df['stock_distributed'] = sub_df['stock_distributed'].copy() + 1

    #     print('identifying p, d, q')
    #     try:
    #         params, params_seasonal = uni.p_d_q(sub_df['stock_distributed'])
    #     except:
    #         params = (0, 1, 1)
    #         params_seasonal = (0, 1, 1, 12)  
        
    #     if len(sub_df):
    #         all_preds.append(predict_SARIMA(sub_df, params, params_seasonal, p.split(' X ')[0], p.split(' X ')[1]))
    #     else:
    #         temp_df = pd.DataFrame([{'date':'2019-07-01', 'prediction': 1, 'month': 7, 'year': 2019, 'product': p.split(' X ')[1], 'site': p.split(' X ')[0]}, {'date':'2019-08-01', 'prediction': 1, 'month': 8, 'year': 2019, 'product': p.split(' X ')[1], 'site': p.split()[0]}, {'date':'2019-09-01', 'prediction': 1, 'month': 9, 'year': 2019, 'product': p.split(' X ')[1], 'site': p.split(' X ')[0]}, {'date':'2019-10-01', 'prediction': 1, 'month': 10, 'year': 2019, 'product': p.split(' X ')[1], 'site': p.split()[0]}])

    #         all_preds.append(temp_df)
    
    # final_df = pd.concat(all_preds, axis = 0)
    # final_df['prediction'] = final_df['prediction'].copy() - 1

    # # making any negative number = 0, since it doesn't make sense to have negative stock
    # final_df['prediction'] = np.where(final_df['prediction']>0, final_df['prediction'], 0)
    # # making the super high outliers the max in the original dataset because it doesn't make sense to have 600,000 products go to one site
    # final_df['prediction'] = np.where(final_df['prediction']<1000, final_df['prediction'], 1728)

    # #creating ID variable to match to the sample submission format
    # final_df['ID']=final_df.year.astype(str)+' X '+final_df.month.astype(str)+' X '+final_df['site']+' X '+final_df['product']

    # # creating csv like the sample submission
    # naive_sub = final_df[['ID','prediction']]
    # naive_sub['prediction'] = naive_sub['prediction'].astype(int)
    # naive_sub.to_csv('results/univariate_submission.csv')