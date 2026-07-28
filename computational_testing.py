#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-dataset testing -- temp file

@author: erinfortin
"""

from algorithms import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import Data
from parameters import *
from evaluator import ModelEvaluator
import metrics


seeds = np.random.choice(10000, size = 1, replace = False)

datasets = pd.read_excel('datasets.xlsx')

ref_point = np.array([1.1, test_data.n + 1])

algorithms = {"replica": replica_FNSGA, 
              "my": my_FNSGA,
              'nsga2': make_pymoo_runner(make_nsga2),
              'spea2': make_pymoo_runner(make_spea2),
              'smsemoa':  make_pymoo_runner(make_smsemoa)}

params = {'LinReg': {**COMMON, **LINREG_PARAMS},
          'LogReg':{**COMMON, **LOGREG_PARAMS},
          'SVC':{**COMMON, **SVC_PARAMS},
          'DT': {**COMMON, **DT_PARAMS}
    }

evaluators = {
    'regression':{
        'LinReg': ModelEvaluator(LinearRegression(), r2_score)
        },
    
    'classification':{
        'LogReg': ModelEvaluator(
            Pipeline([('scaler', StandardScaler()),
                  ('model', LogisticRegression(max_iter=1000))]),
            accuracy_score),
        
        'SVC': ModelEvaluator(
            Pipeline([('scaler', StandardScaler()),
                  ('model', SVC(kernel='rbf'))]), 
            accuracy_score),
        
        'DT': ModelEvaluator(DecisionTreeClassifier(max_depth=5), accuracy_score)
    }
}

results =[]
for _, ds_row in datasets.iterrows():
    dataset_id = ds_row["ID"] 
    dataset_name = ds_row["Name"] 
    task = ds_row["Task"]
    print(f"\n=== {dataset_name} ({task}) ===")
    data = Data(dataset_id)
    
    task_evaluators = evaluators[task]

    for model_name, evaluator in task_evaluators.items():
        print(f"  Model: {model_name}")
        for alg_name, alg_function in algorithms.items():
            print(f"   Algorithm: {alg_name}")
            extreme_1, extreme_2 = metrics.compute_pareto_extremes( data, evaluator)
            metrics_df = metrics.run_metrics(
                seeds, 
                data, 
                evaluator, 
                ref_point, 
                extreme_1, 
                extreme_2, 
                alg_function, 
                params[model_name])
            
            metrics_df["dataset"] = dataset_name
            metrics_df['task'] = task
            metrics_df['model'] = model_name
            metrics_df['algorithm'] = alg_name
            
            results.append(metrics_df)
            
results_df = pd.concat(results, ignore_index=True)
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    