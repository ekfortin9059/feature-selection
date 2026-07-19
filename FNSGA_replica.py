#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:52:45 2026

@author: erinfortin

Feature Non-dominated Sorting Genetic Algorithm (FNSGA)
Replica from paper
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import clone
from population import Population
from data_class import Data
import nsga_functions as nsga

from parameters import COMMON, REPLICA


def replica_FNSGA(data, model, scorer, params, seed=None, plot=False):
    rng = np.random.default_rng(seed)

    N = params["population_size"]
    T = params["generations"]

    p_c = params["crossover_prob"]
    p_m = params["mutation_prob"]

    tournament_param = params["tournament_param"]
    exploration_param = params["exploration_param"]

    seeding_prop = params["seeding_prop"]
    ones_prop = params["ones_prop"]
    ls_param = params["ls_param"]
    
    L = params["L"]
    
    # feature importances from ML model
    feat_importance = SelectFromModel(
        estimator=clone(model)
        ).fit(data.X_train, data.y_train).estimator_.coef_[0]
    
    # initialise population and evaluate
    pop_t = Population(N)
    pop_t.initialize(data.n, feat_importance, seeding_prop, ones_prop, rng)
    
    # initialise archive and store best of population
    pop_t.evaluate(data, model, scorer)
    archive = Population(0)
    archive = nsga.extend_archive(pop_t, archive)
    
    # get prop of individuals coming from each operator
    N_l = round(N * ls_param)
    N_e = N - N_l

    last_fig = None
    last_ax = None
    history = [] if plot else None

    for t in range(T):
        # extract individuals and scores from archive
        A = archive.chromosomes
        model_scores = archive.fitness[:,0] # -R^2
        
        # update feature scores
        denom = A.sum(axis = 0)
        denom[denom==0] = 1
        feat_scores = A.T @ -model_scores
        
        # ensure ranks and crowding distances are set on archive
        fronts = nsga.fast_non_dominated_sort(archive)
        for front in fronts:
            nsga.crowding_distance(front)
    
        # get nondominated set
        nd_set = Population(0)
        nd_set.population = [ind for ind in archive.population if ind.rank == 1]
        
        # evolutionary selection
        evo_pop = nsga.evolutionary_selection(
            archive,
            p_c,
            p_m,
            tournament_param,
            exploration_param,
            len(archive),
            N_e,
            rng
        )
        evo_pop.evaluate(data, model, scorer)

        # local search
        ls_pop = nsga.original_local_search(nd_set, min(L, len(nd_set)), feat_scores, N_l, rng)
        ls_pop.evaluate(data, model, scorer)
        
        # shuffle
        offspring = Population(N_e + N_l)
        offspring.population = [*evo_pop.population, *ls_pop.population]
        rng.shuffle(offspring.population)
        offspring.evaluate(data, model, scorer) # shouldn't add any additional computations since evo and ls evaluated, but added as a fail safe

        # update main archive with this iteration's archive
        archive = nsga.extend_archive(pop_t, archive)
        pop_t = offspring
        
        print(f"Iteration {t+1}/{T} | Archive: {len(archive)} | ND set: {len(nd_set)}")
        
        if plot:
            def _get_fitness(pop):
                f = [ind.fitness for ind in pop.population if ind.fitness is not None]
                return np.array(f) if f else np.empty((0, 2))
            history.append({
                'iteration': t + 1,
                'evo': _get_fitness(evo_pop),
                'ls': _get_fitness(ls_pop)
            })
        
    # get nondominated set to return 
    fronts = nsga.fast_non_dominated_sort(archive)
    final_nd = Population(len(fronts[0]))
    final_nd.population = fronts[0]

    if plot:
        for h in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            for key, label, color, marker in [
                ('evo', 'New solutions: Evolutionary Select', 'red',  'o'),
                ('ls', 'New solutions: Local Search',        'green',   '*'),

            ]:
                d = h[key]
                if len(d) == 0:
                    continue
                ax.scatter(d[:, 1], -d[:, 0], label=label, alpha=0.6,
                           color=color, marker=marker, s=30)
            ax.set_xlabel('Number of Features')
            ax.set_ylabel('R²')
            ax.set_title(f'FNSGA replica\nIteration {h["iteration"]} | Seed {seed}')
            ax.legend(loc='lower right', fontsize=8)
            plt.tight_layout()
            if h["iteration"] == T:
                last_fig = fig
                last_ax = ax
            else:
                plt.show()

    # return final ND set and final iteration plot
    return final_nd, last_fig, last_ax
        
#%%
import time
from pymoo.indicators.hv import HV
 
# Tuned parameters for LinReg from Table 4
dataset_id        = 464
data = Data(dataset_id)

replica_params = {**COMMON, **REPLICA}

seed              = 3444

model = LinearRegression()
scorer = r2_score
 
start = time.time()
result, fig, ax = replica_FNSGA(data, model, scorer, replica_params, seed=seed, plot=True)
print(f"\nSeed: {seed}")
print(f"Runtime: {time.time() - start:.1f}s")
print(f"Final ND set size: {len(result)}")
 
ref_point = np.array([1.1, data.n + 1])
fitness = np.array([ind.fitness for ind in result.population])
hv = HV(ref_point=ref_point)(fitness)
print(f"Hypervolume: {hv:.4f}")
 
ax.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
    edgecolors="black", label = "Final ND Set (from archive)")
ax.legend(loc='lower right', fontsize=8)
ax.set_title(f"FNSGA replica\nFinal Iteration and Pareto Front | Seed {seed}")
plt.show()

    