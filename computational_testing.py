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
import os
import pickle

seeds = np.random.choice(10000, size = 20, replace = False)
datasets = pd.read_excel('datasets.xlsx')


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

# set up to save results throughout process


RESULTS_FILE = 'results_checkpoint.pkl'

# load existing results if resuming after a crash
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)
    print(f"Resuming from checkpoint: {len(results)} results loaded")
else:
    results = []

# build a set of already-completed runs to skip
completed = {
    (r['dataset'], r['model'], r['algorithm'], r['seed'])
    for r in results
}

for _, ds_row in datasets.iterrows():
    dataset_id = ds_row["ID"]
    dataset_name = ds_row["Name"]
    task = ds_row["Task"]
    data = Data(dataset_id)
    task_evaluators = evaluators[task]

    for model_name, evaluator in task_evaluators.items():
        extreme_1, extreme_2 = metrics.compute_pareto_extremes(data, evaluator)
        ref_point = np.array([1.1, data.n + 1])
        
        for alg_name, alg_function in algorithms.items():
            for seed in seeds:
                run_info = (dataset_name, model_name, alg_name, int(seed))
                if run_info in completed:
                    continue  # skip already-done runs
        
        
                print(f"{dataset_name} | {model_name} | {alg_name} | seed {int(seed)}")
                try:
                    metrics_df = metrics.run_metrics(
                        np.array([seed]), data, evaluator,
                        ref_point, extreme_1, extreme_2,
                        alg_function, params[model_name]
                    )
            
                    metrics_df["dataset"] = dataset_name
                    metrics_df['task'] = task
                    metrics_df['model'] = model_name
                    metrics_df['algorithm'] = alg_name
            
                    results.append(metrics_df)
                    completed.add(run_info)
                    
                    with open(RESULTS_FILE, 'wb') as f:
                        pickle.dump(results, f)
                except Exception as e:
                    print(f"  ERROR: {e} — skipping")
            
results_df = pd.concat(results, ignore_index=True)       
results_df.to_csv('results_final.csv', index=False)     

            
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    