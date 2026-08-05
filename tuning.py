#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lets try parameter tuning!
@author: erinfortin
"""

import numpy as np
import pandas as pd
from itertools import product
 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pymoo.indicators.hv import HV
 
from algorithms import my_FNSGA
from data import Data
from evaluator import ModelEvaluator

def evaluate_params(data, evaluator, params, seeds):
    hvs = []
    for seed in seeds:
        try: 
            nd, _,_ = my_FNSGA(data, evaluator, params, seed = int(seed), plot = False)
            F = np.array([ind.fitness for ind in nd.population])
            hvs.append(HV(ref_point=ref_point)(F))
        except Exception as e:
            print(f"   Error on seed {seed}: {e}")
            raise
    return np.mean(hvs), np.std(hvs)

if __name__ == '__main__':
    seed_array = np.random.choice(10000, size = 5, replace = False)
    
    class_data = Data(52)
    reg_data = Data(464)
    tuning_datasets = {
        "LinReg": {"data": reg_data,    "evaluator": ModelEvaluator(LinearRegression(), r2_score)},
        "LogReg": {"data": class_data,  "evaluator": ModelEvaluator(Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000))]), accuracy_score)},
        "SVC":    {"data": class_data,  "evaluator": ModelEvaluator(Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf'))]), accuracy_score)},
        "DT":     {"data": class_data,  "evaluator": ModelEvaluator(DecisionTreeClassifier(max_depth=5), accuracy_score)},
    }
    
    starting_params = {
        "population_size":  100,
        "generations":      20,
        "seeding_prop":     0.3,
        "ones_prop":        0.03,
        "crossover_prob":   0.78,
        "mutation_prob":    0.02,
        "ls_start":         0.20,
        "ls_end":           0.50,
    }
    
    search_space = {
        "mutation_prob":     [0.01, 0.03, 0.05, 0.07, 0.09, 0.1],
        "crossover_prob":    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "seeding_prop":      [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        "ones_prop":         [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        "ls_start":          [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "ls_end":            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
    
    param_names = list(search_space.keys())

    all_combinations = [dict(zip(param_names, values))for values in product(*search_space.values())]
    
    # Remove invalid combinations
    all_combinations = [p for p in all_combinations if p["ls_start"] < p["ls_end"]]
    
    print(f"{len(all_combinations)} valid combinations.")
    
    N_SEARCH = 100

    all_results = []
    final_params = {}
    
    for model_name, config in tuning_datasets.items():
        
        print(f"Tuning for model: {model_name}")
        sampled_params = np.random.choice(len(all_combinations),size=N_SEARCH,replace=False)
        
        data = config["data"]
        evaluator = config["evaluator"]
        ref_point = np.array([0.0, data.n])
        
        best_params = {**starting_params}
    
        best_params = None
        best_mean_hv = -np.inf
        
        for idx in sampled_params:
        
            test_params = starting_params.copy()
            test_params.update(all_combinations[idx])
        
            mean_hv, std_hv = evaluate_params(
                data,
                evaluator,
                test_params,
                seed_array
            )
        
            all_results.append({
                "model": model_name,
                "mean_hv": mean_hv,
                "std_hv": std_hv,
                **test_params
            })
        
            if mean_hv > best_mean_hv:
                best_mean_hv = mean_hv
                best_params = test_params.copy()
        
        final_params[model_name] = best_params
        
        print(f"Best HV: {best_mean_hv:.4f}")
        print(best_params)
    
    
    pd.DataFrame(all_results).to_csv(
    "parameter_search.csv",
    index=False
    )
    
    pd.DataFrame(final_params).T.to_csv(
        "tuned_parameters.csv"
    )
    
    
                
    
