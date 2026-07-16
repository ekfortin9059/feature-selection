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
    
def compute_pareto_extremes(data, eval_model):
    '''
        Computes the two extreme solutions of the feature selection problem 
            1. Best R^2 using all features
            2. Best R^2 using a single feature
    '''
    
    # 1. best r2 using all features
    best_r2_all = eval_model(data, np.ones(data.n))
    extreme_1 = np.array([-best_r2_all, data.n])
    
    # 2. best single feature
    best_r2_single = -np.inf
    for i in range(data.n):
        mask = np.zeros(data.n)
        mask[i] = 1 
        r2 = eval_model(data, mask)
        
        if r2 > best_r2_single:
            best_r2_single = r2
    
    extreme_2 = np.array([-best_r2_single, 1])
    
    return extreme_1, extreme_2

def run_metrics(seeds, N, data, T_max, p_c, p_m, ref_point, 
                   extreme_1, extreme_2, my_nsga2_func, 
                   pymoo_problem, pymoo_algorithm, pymoo_minimize):
    results = []
    
    for seed in seeds:
        # my nsga2 results
        start = time.time()
        pop = my_nsga2_func(N, data, T_max, p_c, p_m, seed = seed)
        my_time = time.time() - start
        
        fronts = fast_non_dominated_sort(pop)
        pareto_front = fronts[0]
        my_F = np.array([ind.fitness for ind in pareto_front])
        my_HV = HV(ref_point=ref_point)(my_F)
        my_spread = compute_spread(my_F, extreme_1, extreme_2)
        my_spacing = compute_spacing(my_F)
        
        # pymoo nsga2 results
        problem = pymoo_problem(data)
        algorithm = pymoo_algorithm(p_c, p_m, N)
        
        start= time.time()
        res = pymoo_minimize(problem, algorithm, termination=('n_gen', T_max),
                             seed = seed, verbose=False)
        pymoo_time = time.time() - start
        
        pymoo_F = res.F
        pymoo_HV = HV(ref_point=ref_point)(pymoo_F)
        pymoo_spread = compute_spread(pymoo_F, extreme_1, extreme_2)
        pymoo_spacing = compute_spacing(pymoo_F)

        results.append({
            'seed': seed,
            'my_HV': my_HV,'pymoo_HV': pymoo_HV,
            'my_spread': my_spread,'pymoo_spread': pymoo_spread,
            'my_spacing': my_spacing, 'pymoo_spacing': pymoo_spacing,
            'my_time': my_time, 'pymoo_time': pymoo_time,
            })
        
    return results
        
        
    
def run_FNSGA_metrics(seeds, data, model, scorer, population_size, generations,
                      crossover_prob, mutation_prob, tournament_param, exploration_param,
                      ref_point, extreme_1, extreme_2,
                      seeding_prop, ones_prop, ls_param, FNSGA_function):

    results = []
    
    for seed in seeds:
        results.append({"seed": seed})
        for val in [False, True]:
            start = time.time()
            # version 1
            nd_front, _, _ = FNSGA_function(
                data, model, scorer,
                population_size, generations,
                crossover_prob, mutation_prob,
                tournament_param, exploration_param,
                seeding_prop, ones_prop, ls_param,
                seed=seed, plot=False, ls_toolbox=val
            )

            end = time.time() - start
            
            F = nd_front.fitness
            seed_idx = np.where(seeds==seed)[0][0]
            results[seed_idx][f"F_v{int(val)}"] = F
            results[seed_idx][f"HV_v{int(val)}"] = HV(ref_point = ref_point)(F)
            results[seed_idx][f"spread_v{int(val)}"] = compute_spread(F, extreme_1, extreme_2)
            results[seed_idx][f"spacing_v{int(val)}"] = compute_spacing(F)
            results[seed_idx][f"time_v{int(val)}"] = end
    return results
            
    
    
    
    
    