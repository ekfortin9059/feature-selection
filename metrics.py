#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:18:41 2026

@author: erinfortin
"""

import time
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from operators import fast_non_dominated_sort

def compute_spacing(F):
    ''' 
    Computes the Efficient Set Spacing of a given front as in Schott (1995).
    '''
    d = np.empty(len(F))
    
    for i in range(len(F)):
        distances = np.sum(np.abs(F[i] - F), axis=1)
        distances[i] = np.inf
        d[i] = np.min(distances)
    
    return (np.sum((d - np.mean(d)) ** 2) / (len(F) - 1))


def compute_spread(F, extreme_1, extreme_2):
    '''
    Computes the spread metric as defined in Deb et al (2002). 
    '''
    
    F_sorted = F[np.argsort(F[:,0])] # sort by first obj
    consecutive_dists = np.linalg.norm(np.diff(F_sorted, axis=0), axis=1)
    d_mean = consecutive_dists.mean()
    
    d_f = np.linalg.norm(F_sorted[0] - extreme_1)
    d_l = np.linalg.norm(F_sorted[-1] - extreme_2)

    num = d_f + d_l + np.sum(np.abs(consecutive_dists - d_mean))
    den = d_f + d_l + (len(consecutive_dists) * d_mean)
    
    return num / den
    
     
def compute_pareto_extremes(data, evaluator):
    '''
    Computes the two extreme solutions of the feature selection problem:
        1. Best score using all features
        2. Best score using a single feature (best individual feature)
    Returns in minimization form: [-score, n_features]
    '''
    # 1. all features
    best_all = evaluator.score(data, np.ones(data.n))
    extreme_1 = np.array([-best_all, data.n])
 
    # 2. best single feature
    best_single = -np.inf
    for i in range(data.n):
        mask = np.zeros(data.n)
        mask[i] = 1
        score = evaluator.score(data, mask)
        if score > best_single:
            best_single = score
    extreme_2 = np.array([-best_single, 1])
 
    return extreme_1, extreme_2   
        
    
def run_metrics(seeds, data, evaluator,ref_point, 
                      extreme_1, extreme_2, algorithm, params):
    results = []
    for seed in seeds:
        seed_idx = np.where(seeds==seed)[0][0]
        print(f"Seed: {seed_idx+1}/{len(seeds)}")
        
        start = time.time()
        nd_front, _, _ = algorithm(
                            data, 
                            evaluator, 
                            params, 
                            seed=seed, 
                            plot=False)
        
        end = time.time() - start
    
        F = np.array([ind.fitness for ind in nd_front.population])
        results.append({
        "seed":        seed,
        "hypervolume": HV(ref_point=ref_point)(F),
        "spread":      compute_spread(F, extreme_1, extreme_2),
        "spacing":     compute_spacing(F),
        "pf_size":     len(F),
        "time":        end,
        })
    return pd.DataFrame(results)
































    
    
    