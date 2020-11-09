### Import necessary dependencies
import numpy as np
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
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objs as go
import clean_dta as clean
import univariate as uni

train, product, site = clean.load_data('../data/')

df = clean.clean_data(train, product, site)

df1, monthly = clean.group_and_subset_data(df, ['calendar'])

Total_stock_distributed =df.groupby(['month','district','site_code']).product_code.sum().reset_index()
Total_stock_distributed.rename({'product_code':'Total_product_on_month_basis_for_each_site_code'}, axis=1, inplace=True)

def count_product(text):
    count = sum([1 for char in text if char in str.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

Total_stock_distributed['Total_product_persite'] = Total_stock_distributed['Total_product_on_month_basis_for_each_site_code'].apply(lambda x: len(x) - x.count(" "))

Total_stock_distributed['Total_product_persite']= Total_stock_distributed['Total_product_persite']/7
Total_stock_distributed['Total_product_persite']=Total_stock_distributed['Total_product_persite'].astype(int)

Total_stock_distributed['Total_product']=Total_stock_distributed['Total_product_on_month_basis_for_each_site_code'].map(Total_stock_distributed['Total_product_on_month_basis_for_each_site_code'].value_counts())
