#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:53:16 2026

@author: erinfortin
"""
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
import pandas as pd

class Data:
    def __init__(self, _id):
        if _id == 'energy':
            df = pd.read_csv("large-scale+wave+energy+farm/WEC_Perth_49.csv")
            X = df.drop('Total_Power', axis = 1) 
            X = df.drop(columns=df.filter(regex=r'^Power\d+').columns)
            y = df['Total_Power']
        else:   
            data_obj = fetch_ucirepo(id=_id)
            X = data_obj.data.features.dropna(axis=1)
            X = X.select_dtypes(exclude = ['object'])
            y = data_obj.data.targets
        
        valid_indices = y.dropna().index

        self.X = X.loc[valid_indices]
        self.y = y.loc[valid_indices]
        self.n = len(self.X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            
            