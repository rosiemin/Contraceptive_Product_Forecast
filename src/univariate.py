import numpy as np
import pandas as pd
import calendar
import datetime
from dateutil.relativedelta import relativedelta
import itertools
import clean_dta as clean
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge,Lasso,LassoLarsCV
from  scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
from fbprophet import Prophet


from statsmodels.tsa.ar_model import AutoReg
from random import random
import seaborn as sns

from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot') # I also like fivethirtyeight'
matplotlib.rcParams.update({'font.size': 16, 'font.family': 'sans'})

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def MAE(y_test,y_pred):
    return mean_absolute_error(y_test, y_pred)

def MAPE(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def MASE(training_df, y_test, y_pred):
    n = training_df.shape[0]
    d = np.abs(  np.diff( training_df) ).sum()/(n-1)
    
    errors = np.abs(y_test - y_pred )
    return errors.mean()/d

def train_test_split(df, split_pct = 0.8):
    train_n = int(len(df) * split_pct)

    train_df = df.iloc[:train_n, :]
    test_df = df.iloc[train_n: , :]

    return train_df, test_df

def test_stationarity(timeseries, dates, plot_flag = False):
    rolmean = timeseries.rolling(window=3).mean()
    rolstd = timeseries.rolling(window=3).std()
    
    if plot_flag:
        plt.figure(figsize=(14,5))
        sns.despine(left=True)
        plt.plot(dates, timeseries, color='blue',label='Original')
        plt.plot(dates, rolmean, color='red', label='Rolling Mean')
        plt.plot(dates, rolstd, color='black', label = 'Rolling Std')

        plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def p_d_q(y):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


    min_aic = 1000
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            
            mod = sm.tsa.statespace.SARIMAX(y,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False, disp = 0)

            results = mod.fit()
            
            if results.aic < min_aic:
                min_aic = results.aic
                param_use, param_seasonal_use = param, param_seasonal

    print('ARIMA{}x{}12 - AIC:{}'.format(param_use, param_seasonal_use, min_aic))

    return param_use, param_seasonal_use

def SARIMA(df, start, order, seasonal_order, test_df, label= 'SARIMA', plot_flag = False):

    results = sm.tsa.statespace.SARIMAX(df['stock_distributed'],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False).fit(use_boxcox=True)

    # print(results.summary().tables[1])    

    pred = results.get_prediction(start=start, dynamic=False)
    pred_ci = pred.conf_int()
    if plot_flag:
        ax = df.plot(label='observed',x = 'calendar')
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date', fontsize = 14)
        ax.set_ylabel('Stock Distribution', fontsize = 14)
        ax.tick_params(labelsize = 12)
        plt.title('SARIMA')
        plt.legend()
        plt.show()

    test_df[label] = pred.predicted_mean

    return test_df

def holt_winters(df, seasonal_period, test_df, label = 'HW', plot_flag = False):
    fit2 = ExponentialSmoothing(np.asarray(df['stock_distributed']) ,seasonal_periods=seasonal_period ,trend='add', seasonal='add').fit(use_boxcox=True)
    test_df[label] = fit2.forecast(len(test_df))

    if plot_flag:
        plt.figure(figsize=(16,8))
        plt.plot(df_use['stock_distributed'], label='Actual')
        plt.plot(test_df[label], label=label)
        plt.legend(loc='best')
        plt.title('Holt-Winter Exponential Smoothing')
        plt.show()

    return test_df

def fb_prophet(df, df_test, period = 6, label = 'fbProphet', plot_flag = False):
    df_use = df.rename(columns={'calendar': 'ds', 'stock_distributed': 'y'})
    model = Prophet()
    model.fit(df_use)

    forecast = model.make_future_dataframe(periods=period, freq='MS')
    forecast = model.predict(forecast)

    if plot_flag:
        plt.figure(figsize=(18, 6))
        model.plot(forecast, xlabel = 'Date', ylabel = 'Stock Distributed')
        plt.title('FB Prophet')
        plt.show()

    forecast_trunc = forecast[len(forecast)-len(df_test):].copy()
    forecast_trunc.index = df_test.index

    df_test[label] = forecast_trunc.yhat
    

    return df_test


def cross_val(test_df, train_df, test_lst, label_lst):
    metrics = pd.DataFrame(columns = ['label', 'RMSE', 'MAE', 'MASE'])
    y_test = test_df['stock_distributed']
    y_train = train_df['stock_distributed']
    for test, label in zip(test_lst, label_lst):
        metrics.loc[len(metrics)] = [label, round(RMSE(y_test, test_df[test]), 2), round(MAE(y_test, test_df[test]), 2), round(MASE(y_train, y_test, test_df[test]), 2)]
    
    return metrics

if __name__ == "__main__":
    train, product, site = clean.load_data('../data/')

    df = clean.clean_data(train, product, site)

    df = clean.remove_short_df(df)

    all_df, df_use = clean.group_and_subset_data(df, ['calendar'])

    train, test = train_test_split(df_use, 0.8)

    test_stationarity(df_use.stock_distributed.dropna(), df_use.calendar, True)

    params, params_seasonal = p_d_q(df_use['stock_distributed'])

    test = SARIMA(df_use, '2018-10-01', params, params_seasonal, test, 'SARIMA', True)

    test = holt_winters(df_use, 3, test, 'HW-3', True)

    test = fb_prophet(train, test, len(test), plot_flag = True)

    metrics = cross_val(test, train, ['SARIMA', 'HW-3', 'fbProphet'], ['SARIMA', 'Holt-Winters 14', 'FB Prophet'])