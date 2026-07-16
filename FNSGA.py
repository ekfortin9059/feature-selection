#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:52:45 2026

@author: erinfortin

Feature Non-dominated Sorting Genetic Algorithm (FNSGA)
Faithful implementation of Algorithm 1 
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


def FNSGA(data, model, scorer, N, T_max, crossover_prob, mutation_prob,
          tournament_param, exploration_param,
          seeding_prop, ones_prop,
          ls_param, L,
          seed=None, plot=False):
    
    rng = np.random.default_rng(seed)
    
    # feature importances from ML model
    feat_importance = SelectFromModel(
        estimator= clone(model)
        ).fit(data.X, data.y).estimator_.coef_[0]
    
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

    history = [] if plot else None

    for t in range(T_max):
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
            crossover_prob,
            mutation_prob,
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
        
        print(f"Iteration {t+1}/{T_max} | Archive: {len(archive)} | ND set: {len(nd_set)}")
        
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
        last_fig = None
        last_ax = None
        for h in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            for key, label, color, marker in [
                ('evo', 'New solutions: Evolutionary Select', 'red',  'o'),
                ('ls', 'New solutions: Local Search',        'green',   'o'),

            ]:
                d = h[key]
                if len(d) == 0:
                    continue
                ax.scatter(d[:, 1], -d[:, 0], label=label, alpha=0.6,
                           color=color, marker=marker, s=30)
            ax.set_xlabel('Number of Features')
            ax.set_ylabel('R²')
            ax.set_title(f'Iteration {h["iteration"]}')
            ax.legend(loc='lower right', fontsize=8)
            plt.tight_layout()
            if h["iteration"] == T_max:
                last_fig = fig
                last_ax = ax
            else:
                plt.show()

    # return final ND set and final iteration plot
    return final_nd, last_fig, last_ax
        

if __name__ == '__main__':
    import time
    from pymoo.indicators.hv import HV
 
    # Tuned parameters for LinReg from Table 4
    dataset_id        = 464
    N                 = 100
    T_max             = 20
    crossover_prob    = 0.78
    mutation_prob     = 0.02
    tournament_param  = 0.59
    exploration_param = 0.63
    seeding_prop = 0.4
    ones_prop = 0.01
    ls_param          = 0.5
    L                 = 20
    seed              = np.random.choice(10000, size = 1, replace = False)
    model = LinearRegression()
    scorer = r2_score
    data = Data(dataset_id)
 
    start = time.time()
    result, fig, ax = FNSGA(
        data, model, scorer, 
        N, T_max,
        crossover_prob, mutation_prob,
        tournament_param, exploration_param,
        seeding_prop, ones_prop,
        ls_param, L,
        seed=seed, plot=True
    )
    print(f"\nSeed: {seed}")
    print(f"Runtime: {time.time() - start:.1f}s")
    print(f"Final ND set size: {len(result)}")
 
    ref_point = np.array([1.1, data.n + 1])
    fitness = np.array([ind.fitness for ind in result.population])
    hv = HV(ref_point=ref_point)(fitness)
    print(f"Hypervolume: {hv:.4f}")
 
    ax.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
        edgecolors="blue", label = "Final ND Set (from archive)")
    ax.legend(loc='lower right', fontsize=8)
    plt.show()





    