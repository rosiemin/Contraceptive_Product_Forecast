# Data Processing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

# Model Selection
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import yellowbrick as yb 
from yellowbrick.regressor import prediction_error, residuals_plot
from yellowbrick.regressor.alphas import AlphaSelection

# Regressor models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import calendar
import datetime
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


site_products = dev_data[['site_code','product_code']]
site_products_nonpooled_models = pd.DataFrame()

for index, data in tqdm(site_products.iterrows()):
    # Filter data to site, product pair
    site_key = data['site_code']
    product_key = data['product_code']
    
    logical_mask =  (dev_data['site_code']==site_key) & (dev_data['product_code']==product_key)
    site_product_level_data_ = dev_data.loc[logical_mask,:]
    
    coded_cols = ['product_code','calendar','region','site_code']

    X = site_product_level_data_.drop(columns = ['stock_distributed','out_of_stock_during_period']+coded_cols)
    
    #X = site_product_level_data_.loc[:, ['stock_ordered','year_month','out_of_stock_during_period']]
    y = site_product_level_data_['stock_distributed']
    
    if len(X) > 5:
        # Impute missing values in the dataset by incorporating information from the other features
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(X)

        # the model learns that the second feature is double the first
        X_imputed = pd.DataFrame(np.round(imp.transform(X)),columns =X.columns)

        # Reset the y matrix index so it matches with the new X
        y = y.reset_index().drop(columns='index')
        
        # Split the data set in to training and test sets
        X_train, X_test, y_train, y_test  = train_test_split( X_imputed, y, test_size=0.2, random_state=42)

        # non_pooled regression
        non_pooled = sm.OLS(y_train, X_train).fit()
        y_pred = non_pooled.predict(X_test)
        r2, rmse = metrics.r2_score(y_test, y_pred), np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mase = MASE(y_train,y_test,y_pred)
    else:
        rmse,mase = None, None
    
    data = {"site_code" :site_key, "product_code" : product_key,"R2" : [r2],"RMSE" : [rmse],"MASE":[mase],'Obs':len(X)}
    record_df = pd.DataFrame(data)
    site_products_nonpooled_models = site_products_nonpooled_models.append(record_df)