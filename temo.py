#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 09:57:21 2026

@author: erinfortin
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle
 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pymoo.indicators.hv import HV
from scipy.stats import friedmanchisquare
 
from algorithms import my_FNSGA, replica_FNSGA
from data import Data
from parameters import *
from evaluator import ModelEvaluator
import metrics
from population import Population
from individual import Individual
from data import Data
import operators as op

seed = 3444
rng = np.random.default_rng(seed)
data = Data(360)

params = {**LINREG_PARAMS, **COMMON}
N = params["population_size"]
T = params["generations"]
p_c = params["crossover_prob"]
p_m = params["mutation_prob"]
tournament_param = params["tournament_param"]
exploration_param = params["exploration_param"]
seeding_prop = params["seeding_prop"]
ones_prop = params["ones_prop"]
ls_param = params["ls_param"]

evaluator = ModelEvaluator(LinearRegression(), r2_score)

ref_point = np.array([-0.05, data.n + 1])
ext_1, ext_2 = metrics.compute_pareto_extremes(data, evaluator)


feat_importances = evaluator.feature_importances(data)

# generate initial population (including rank and crowding distance)
pop_t = Population()
pop_t.initialise(data.n, N, feat_importances, seeding_prop, ones_prop ,rng)
pop_t.evaluate(data, evaluator)

fronts = op.fast_non_dominated_sort(pop_t)
for front in fronts:
    op.crowding_distance(front)

fitness = np.array([ind.fitness for ind in pop_t.population])
ax2 = plt.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
    edgecolors="black", label = "pop")
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))

test1 = op.evolutionary_selection2(pop_t, p_c, p_m, 50, rng)
test1.evaluate(data, evaluator)
fitness1 = np.array([ind.fitness for ind in test1.population])
ax.scatter(fitness1[:, 1], -fitness1[:, 0],facecolors="none",
    edgecolors="blue", label = "new EVO")

test2 = op.evolutionary_selection(pop_t, p_c, p_m, tournament_param, exploration_param, 100, 50, rng)
test2.evaluate(data, evaluator)
fitness2 = np.array([ind.fitness for ind in test2.population])
ax.scatter(fitness2[:, 1], -fitness2[:, 0],facecolors="none",
    edgecolors="red", label = "old EVO")
plt.show()