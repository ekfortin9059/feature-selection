#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:40:21 2026

@author: erinfortin
"""
import numpy as np 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from population import Population
from individual import Individual
from data_class import Data
import nsga_functions as nsga
import matplotlib.pyplot as plt
import benchmark_metrics as bm
from ML_code import eval_model
from sklearn.metrics import r2_score
#%%
dataset_id = 464
population_size = 100
archive_size = 100
generations = 20

crossover_prob = 0.8
mutation_prob = 0.07

tournament_param = 0.6
exploration_param = 0.5

seeding_prop = 0.4
ones_prop = 0.01

ls_param = 0.5

data = Data(dataset_id)

seed = np.random.choice(10000, size = 1, replace = False)

rng = np.random.default_rng(seed)
#%% 
# feature importances based on model evaluation
feat_importance = SelectFromModel(
    estimator=LinearRegression()
    ).fit(data.X_train, data.y_train).estimator_.coef_[0]

 #%%
# generate initial population
pop_t = Population()
pop_t.initialise(data.n, population_size, feat_importance, seeding_prop, ones_prop, rng)
pop_t.evaluate(data, LinearRegression(), r2_score)

fronts = nsga.fast_non_dominated_sort(pop_t)
for front in fronts:
    nsga.crowding_distance(front)

offspring = nsga.evolutionary_selection(pop_t, crossover_prob, mutation_prob, tournament_param, exploration_param, population_size, 25, rng)
offspring.evaluate(data, LinearRegression(), r2_score)

#%%

small = Population()
small.initialise(data.n, 25, feat_importance, seeding_prop, ones_prop, rng)
small.evaluate(data, LinearRegression(), r2_score) 
#%% 
pop_t.update_pop(offspring, population_size, data.n, rng)

#%% 
A = pop_t.chromosomes
model_scores = pop_t.fitness[:, 0]
denom = A.sum(axis=0)
denom[denom == 0] = 1
feat_scores = (A.T @ -model_scores) / denom

nd_set = Population()
nd_set.population = [ind for ind in pop_t.population if ind.rank == 1]

N_l = round(population_size * ls_param)
LS_return = N_l // 4

# ls1 = nsga.add_local_search(nd_set, feat_scores, LS_return, rng)
# ls2 = nsga.remove_local_search(nd_set, feat_scores, LS_return, rng)
ls3 = nsga.merge_local_search(nd_set, round(N_l - LS_return*3), rng)
# ls4 = nsga.add_remove_local_search(nd_set, feat_scores, round(N_l - LS_return*3), rng)

#%% iteration
fronts = nsga.fast_non_dominated_sort(pop_t)
for front in fronts:
    nsga.crowding_distance(front)

# extract non-dominated set from archive
nd_archive = Population(0)
nd_archive.population = [ind for ind in archive.population if ind.rank == 1]

# get feature scores of nondominated solutions (for local search)
A = nd_archive.chromosomes
model_scores = nd_archive.fitness[:, 0]
denom = A.sum(axis=0)
denom[denom == 0] = 1


#%%

import numpy as np 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from population import Population
from individual import Individual
from data_class import Data
import nsga_functions as nsga
import matplotlib.pyplot as plt

dataset_id = 464
population_size = 100
archive_size = 100
generations = 20

crossover_prob = 0.8
mutation_prob = 0.07

tournament_param = 0.6
exploration_param = 0.5

seeding_prop = 0.4
ones_prop = 0.01

ls_param = 0.5

seed = 123
data = Data(dataset_id)

def FNSGA1(data, population_size, generations, crossover_prob, mutation_prob,
          tournament_param, exploration_param,
          seeding_prop, ones_prop, ls_param,
          seed=None, plot=False):
    
    rng = np.random.default_rng(seed)
     
    # feature importances based on model evaluation
    feat_importance = SelectFromModel(
        estimator=LinearRegression()
        ).fit(data.X, data.y).estimator_.coef_[0]
     
    # generate initial population
    pop_t = Population(population_size)
    pop_t.initialize(data.n, feat_importance, seeding_prop, ones_prop, rng)
    pop_t.evaluate(data, LinearRegression(), r2_score)
     
    # store initial pop in archive 
    archive = Population(population_size)
    archive = nsga.update_archive(pop_t, archive, population_size)
    
    # amount of individuals for offspring coming from LS vs ES
    N_l = round(population_size * ls_param)
    N_e = population_size - N_l
     
    for _ in range(generations):
        # set ranks/crowding on archive (needed for ND extraction and tournament select)
        fronts = nsga.fast_non_dominated_sort(archive)
        for front in fronts:
            nsga.crowding_distance(front)
    
        # set ranks/crowding on pop_t (needed for tournament select)
        fronts_t = nsga.fast_non_dominated_sort(pop_t)
        for front in fronts_t:
            nsga.crowding_distance(front)
        # extract non-dominated set from archive
        nd_archive = Population(0)
        nd_archive.population = [ind for ind in archive.population if ind.rank == 1]
        
        # get feature scores of nondominated solutions (for local search)
        A = nd_archive.chromosomes
        model_scores = nd_archive.fitness[:, 0]
        denom = A.sum(axis=0)
        denom[denom == 0] = 1
        feat_scores = (A.T @ -model_scores) / denom
        
        # Evolutionary Selection
        evo_pop = nsga.evolutionary_selection(
            pop_t,                 
            crossover_prob,
            mutation_prob,
            tournament_param,
            exploration_param,
            len(pop_t),
            N_e,
            rng
        )
        evo_pop.evaluate(data,LinearRegression(), r2_score)
        
        temp_pop = Population(population_size)
        
        # Local Search 1 - find nearby solutions to ND solutions  
        temp_pop.population.extend(nsga.original_local_search(nd_archive, len(nd_archive), feat_scores, round(N_l * 0.2), rng).population)
        
        # Local Search 2 - add
        temp_pop.population.extend(nsga.add_local_search(
            nd_archive, feat_scores, round(N_l * 0.2), rng
        ).population)
        
        # Local Search 3 - remove
        temp_pop.population.extend(nsga.remove_local_search(
            nd_archive, feat_scores, round(N_l * 0.2), rng
        ).population)
        
        # Local Search 4 - merge
        temp_pop.population.extend(nsga.merge_local_search(
            nd_archive, round(N_l * 0.2), rng
        ).population)
        
        # Local Search 5 - add-remove
        temp_pop.population.extend(nsga.add_remove_local_search(
            nd_archive, feat_scores, round(N_l - len(temp_pop)), rng
        ).population)
        
        
        # shuffle solutions 
        offspring = Population(N_e + N_l)
        offspring.population = [*evo_pop.population, *temp_pop.population]
        rng.shuffle(offspring.population)   
         
        # update the archive with new best solutions
        archive = nsga.update_archive(offspring, archive, population_size)
        
        # update current population to be the offspring
        pop_t = offspring
     
    # final ND solution set 
    fronts = nsga.fast_non_dominated_sort(archive)
    final_nd = Population(len(fronts[0]))
    final_nd.population = fronts[0]
 
    return final_nd


f = FNSGA1(data, population_size, generations, crossover_prob, mutation_prob,
          tournament_param, exploration_param,
          seeding_prop, ones_prop, ls_param,
          seed=123, plot=False)
