import numpy as np
import pandas as pd
import calendar
import datetime
from dateutil.relativedelta import relativedelta
import category_encoders as ce
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso 
from xgboost import XGBRegressor
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

train, product, site = clean.load_data('data/')

df = clean.clean_data(train, product, site)

stock_distribution_monthly = df.groupby(['calendar']).stock_distributed.sum().reset_index()
