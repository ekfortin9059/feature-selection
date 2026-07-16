#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:32:57 2026

@author: erinfortin
"""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score


# =============================================================================
# Model Evaluation function for selected features (Linear Regression)
# =============================================================================
def eval_model(data, F, model, scorer):
    mask = np.array(F).flatten().astype(bool)
    cols = data.X_train.columns[mask]
    M = clone(model)
    M.fit(data.X_train[cols], data.y_train)
    y_pred = M.predict(data.X_test[cols])
    return scorer(data.y_test.values.ravel(), y_pred)