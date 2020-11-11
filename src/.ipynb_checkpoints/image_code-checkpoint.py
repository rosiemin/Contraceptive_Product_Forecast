import plotly.graph_objs as go
import chart_studio.plotly as py
import pandas as pd
import plotly
import chart_studio
import src.clean_dta as clean
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

import matplotlib.gridspec as gridspec
chart_studio.tools.set_credentials_file(username='rosiemin', api_key='PuHQaQ7Z4wsNDGSzVKKV')
def hist_dat(df,col,colr,state, show = True, path = None):
    textstr = '\n'.join((
    r'Max: {}'.format(df[col].max()),
    r'Minimum: {}'.format(df[col].min()),
    r'Mean: {}'.format(round(df[col].mean(),2)),
    r'Total N: {}'.format(len(df[col]))))


    f, axes = plt.subplots(2, 1, figsize=(20,10))
    sns.distplot(df[col], ax=axes[1], color = colr)
    sns.boxplot(df[col], ax=axes[0], color = colr, boxprops=dict(alpha=.5)).set_xlabel('')
    axes[0].set_xticklabels(labels='')
    # f.subplots_adjust(top=0.8)

    plt.suptitle(f"Distribution of {col} {state} cleaning data", fontsize = 20)
    plt.xlabel(f"{col}", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5,prop=dict(fontsize=16))
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    f.subplots_adjust(wspace=0, hspace=0)

    if show:
        plt.show()
    if path:
        plt.savefig(path)