import numpy as np
import pandas as pd
import calendar
import datetime
from dateutil.relativedelta import relativedelta
import itertools

from random import random
import seaborn as sns


def load_data(BASE_PATH):
    train = pd.read_csv(BASE_PATH + 'Train.csv')
    train.drop('stock_stockout_days', axis = 1, inplace = True)

    product = pd.read_csv(BASE_PATH + 'product.csv')
    site_df = pd.read_csv(BASE_PATH + 'service_delivery_site_data.csv')

    return train, product, site_df

def clean_data(t, p, s):
    t=t.merge(s[['site_code','site_type','site_latitude','site_longitude']],on='site_code')
    t=t.merge(p[['product_code','product_type']],on='product_code')

    t['ID']=t.year.astype(str)+' X '+t.month.astype(str)+' X '+t['site_code']+' X '+t['product_code']

    t['calendar'] = pd.to_datetime(t['year'].astype(str)  + t['month'].astype(str), format='%Y%m')

    t['site_product'] = t['site_code'] + ' X ' + t['product_code']

    return t

def group_and_subset_data(df, groupby_lst):
    df = df.sort_values(groupby_lst)
    use_df_Y = df.groupby(groupby_lst).sum().stock_distributed.reset_index()
    use_df = df.groupby(groupby_lst).sum().reset_index()

    use_df_Y.set_index('calendar', inplace = True, drop = False)
    
    return use_df, use_df_Y

def remove_short_df(df):
    product_names = df['product_code'].unique()
    site_names = df['site_code'].unique()

    all_names = [s + ' ' + p for s in site_names for p in product_names]

    df_keep = []
    site_product = []
    
    for p in all_names:
        # pdb.set_trace()
        sub_df = df[(df['product_code']==p.split()[1]) & (df['site_code']==p.split()[0])]  
        if sub_df.year.max()<2019 or len(sub_df)<6:
            site_product.append(p)
        else:
            df_keep.append(sub_df)
    
    df_new = pd.concat(df_keep, axis = 0)
    return df_new, site_product


if __name__ == "__main__":
    pass