# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:02:15 2017

@author: Michal
"""

import pandas as pd

data = pd.read_csv('C:/Users/Michal/Desktop/Publico/Dane_CSV/AllArticleProvidersGood.csv',sep=';', error_bad_lines=False)
data.columns = ['ID','Name','Provider']
data_short = data[244277:]

data_short = data_short.replace('mobile','android')
    
data=data.drop(data.index[244277:386761])

frames = [data,data_short]
data = pd.concat(frames)
SUM = []
SUM = data.groupby(['Name','Provider']).count()

SUM.to_csv('pobrania_android_ios.csv',sep=';')