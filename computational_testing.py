#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-dataset testing 

@author: erinfortin
"""

from algorithms import *
import numpy as np
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
from joblib import Parallel, delayed

def run_single_task(task_tuple, data, evaluator, 
                    ref_point, extreme_1, extreme_2, param_dict):
    
    dataset_name, model_name, alg_name, alg_function, seed, task = task_tuple
    
    output_file = f"checkpoints/run_{dataset_name}_{model_name}_{alg_name}_{seed}.pkl"
    if os.path.exists(output_file):
        return None
    
    try:
        np.random.seed(seed)
                
        metrics_df = metrics.run_metrics(
            np.array([seed]), data, evaluator,
            ref_point, extreme_1, extreme_2,
            alg_function, param_dict
        )
        metrics_df["dataset"] = dataset_name
        metrics_df['task'] = task
        metrics_df['model'] = model_name
        metrics_df['algorithm'] = alg_name
        
        with open(output_file, 'wb') as f:
            pickle.dump(metrics_df, f)
            
        return metrics_df
    
    except Exception as e:
        print(f"ERROR on {dataset_name} | {model_name} | {alg_name} | seed {seed}: {e}")
        return None


if __name__ == '__main__':
    
    os.makedirs('checkpoints', exist_ok=True)
    
    #configurations
    seeds = np.random.choice(10000, size = 20, replace = False)
    datasets = pd.read_excel('datasets.xlsx')

    algorithms = {"replica": replica_FNSGA, 
                  "my": my_FNSGA,
                  'nsga2': make_pymoo_runner(make_nsga2),
                  'spea2': make_pymoo_runner(make_spea2),
                  'moead': make_pymoo_runner(make_moead),
                  'smsemoa':  make_pymoo_runner(make_smsemoa),
                  'dnsga2':  make_pymoo_runner(make_dnsga2),
                  'nsde':  make_pymoo_runner(make_nsde)
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

    all_tasks = []
    dataset_cache = {}
    
    print("Loading data and evaluating pareto extremes and ref point")
    
    for _, ds_row in datasets.iterrows():
        dataset_id = ds_row["ID"]
        dataset_name = ds_row["Name"]
        task = ds_row["Task"]
        task_evaluators = evaluators[task] # classification or regression data set
        
        #save previous evaluations of loading data and calculating extreme/ref points
        if dataset_id not in dataset_cache:
            print(f"Importing dataset {dataset_name}")
            data = Data(dataset_id)
            cached_evaluations = {}
            
            for model_name, evaluator in task_evaluators.items():
                ext_1, ext_2 = metrics.compute_pareto_extremes(data, evaluator)
                ref_pt = np.array([0.0, data.n])  # worst solution for the form [-score, n_feat]
                
                cached_evaluations[model_name] = {
                    "evaluator": evaluator,
                    "extreme_1": ext_1,
                    "extreme_2": ext_2,
                    "ref_point": ref_pt
                    }
            dataset_cache[dataset_id] = {"data": data, "evaluations": cached_evaluations}
            
            for model_name in task_evaluators.keys():
                for alg_name, alg_function in algorithms.items():
                    param_dict = get_algorithm_params(alg_name, model_name)
                    for seed in seeds:
                        all_tasks.append((dataset_name, model_name, alg_name, alg_function, seed, task, dataset_id, param_dict))
    
    print("Completed data loading")
                   
    tasks_to_run = []
    for task_info in all_tasks:
        dataset_name, model_name, alg_name, _, seed, _, _, _ = task_info         
        chk_path = f"checkpoints/run_{dataset_name}_{model_name}_{alg_name}_{seed}.pkl"
        if not os.path.exists(chk_path):
            tasks_to_run.append(task_info)
    
    # multi-run execution using Parallel processes      
    print(f"Total experiment matrix setup: {len(all_tasks)} runs.")
    print(f"Remaining tasks to evaluate: {len(tasks_to_run)} runs.")

    if len(tasks_to_run) > 0:
        print("\nStarting parallel experiment runs")
        Parallel(n_jobs = -2, verbose = 10)(
            delayed(run_single_task)(
                task_tuple=(dataset_name, model_name, alg_name, alg_function, seed, task),
                data=dataset_cache[dataset_id]["data"],
                evaluator=dataset_cache[dataset_id]["evaluations"][model_name]["evaluator"],
                ref_point=dataset_cache[dataset_id]["evaluations"][model_name]["ref_point"],
                extreme_1=dataset_cache[dataset_id]["evaluations"][model_name]["extreme_1"],
                extreme_2=dataset_cache[dataset_id]["evaluations"][model_name]["extreme_2"],
                param_dict=param_dict, 
            )
            for dataset_name, model_name, alg_name, alg_function, seed, task, dataset_id, param_dict in tasks_to_run
        )
    else:
        print("\nAll tasks completed")
    
    compiled_results = []
    
    for task_info in all_tasks:
        dataset_name, model_name, alg_name, _, seed, _, _ = task_info
        chk_path = f"checkpoints/run_{dataset_name}_{model_name}_{alg_name}_{seed}.pkl"
        if os.path.exists(chk_path):
            with open(chk_path, 'rb') as f:
                compiled_results.append(pickle.load(f))
        
    if compiled_results:
        results_df = pd.concat(compiled_results, ignore_index=True)
        results_df.to_csv('results_final.csv', index=False)
    
            