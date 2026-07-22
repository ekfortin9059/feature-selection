#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:18:41 2026

@author: erinfortin
"""

import time
import numpy as np
from pymoo.indicators.hv import HV
from nsga_functions import fast_non_dominated_sort

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
    
def compute_pareto_extremes(data, eval_model, model, scorer):
    '''
        Computes the two extreme solutions of the feature selection problem 
            1. Best R^2 using all features
            2. Best R^2 using a single feature
    '''
    
    # 1. best r2 using all features
    best_r2_all = eval_model(data, np.ones(data.n), model, scorer)
    extreme_1 = np.array([-best_r2_all, data.n])
    
    # 2. best single feature
    best_r2_single = -np.inf
    for i in range(data.n):
        mask = np.zeros(data.n)
        mask[i] = 1 
        r2 = eval_model(data, mask, model, scorer)
        
        if r2 > best_r2_single:
            best_r2_single = r2
    
    extreme_2 = np.array([-best_r2_single, 1])
    
    return extreme_1, extreme_2
        
        
    
def run_FNSGA_metrics(seeds, data, model, scorer,
                      ref_point, extreme_1, extreme_2, 
                      algorithms, params):
    results = []
    print("Running metrics")
    for seed in seeds:
        print(f"Seed: {seed}")
        results.append({"seed": seed})
        for alg in [*algorithms]:
            start = time.time()
            nd_front, _, _ = algorithms[alg](data, model, scorer, params[alg], seed=seed, plot=False)
            
    
            end = time.time() - start
            F = nd_front.fitness
            seed_idx = np.where(seeds==seed)[0][0]
            results[seed_idx][f"F_{alg}"] = F
            results[seed_idx][f"HV_{alg}"] = HV(ref_point = ref_point)(F)
            results[seed_idx][f"spread_{alg}"] = compute_spread(F, extreme_1, extreme_2)
            results[seed_idx][f"spacing_{alg}"] = compute_spacing(F)
            results[seed_idx][f"time_{alg}"] = end
    return results



    
    
    