#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:32:57 2026

@author: erinfortin
"""
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# =============================================================================
# Model Evaluation class
# =============================================================================

class ModelEvaluator:
    def __init__(self, model, scoring_func):
        self.model = model
        self.scoring_func = scoring_func
        
    def score(self, data, chromosome):
        mask = np.array(chromosome).flatten().astype(bool)
        if not mask.any():
            return -np.inf
        
        if hasattr(data.X_train, 'columns'):
            X_train = data.X_train.loc[:, mask]
            X_test = data.X_test.loc[:, mask]
        else:
            X_train = data.X_train[:, mask]
            X_test = data.X_test[:, mask]
        y_train = data.y_train.values.ravel()
        y_test = data.y_test.values.ravel()
        
        M = clone(self.model)
        M.fit(X_train, y_train)
        y_pred = M.predict(X_test)
        return self.scoring_func(y_test, y_pred)
    
    def fitness(self, data, chromosome):
        return [-self.score(data,chromosome), np.array(chromosome).sum()]
    
    def feature_importances(self, data):
        model = clone(self.model)
        model.fit(data.X_train, data.y_train.values.ravel())
        
        if isinstance(model, Pipeline):
            estimator = model[-1]
        else:
            estimator = model  
            
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        
        return np.ones(data.n)
        
        
    def feature_importances_sfm(self, data):
        model = clone(self.model)
        model.fit(data.X_train, data.y_train.values.ravel())
        
        selector = SelectFromModel(estimator = model, prefit=True)
        
        estimator = selector.estimator
        
        if isinstance(estimator, Pipeline):
            estimator = estimator[-1]
            
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_

        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        
        return np.ones(data.n)
            