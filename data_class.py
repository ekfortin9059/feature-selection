#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:53:16 2026

@author: erinfortin
"""
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, _id):
        data_obj = fetch_ucirepo(id=_id)
        self.X = data_obj.data.features
        self.y = data_obj.data.targets
        self.n = len(self.X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)