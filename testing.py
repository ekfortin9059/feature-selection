#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run and compare algorithms

0. Preliminaries
1. Single-run and plotting of improved FNSGA
2. Single-run and plotting of improved FNSGA
3. Multi-seed run of both algorithms
4. Multi-algorithm comparisons (incl. Pymoo functions) --- TO DO
"""
# =============================================================================
# 0. Preliminaries
# =============================================================================
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

#%%
# seed and data

seed = 3444
test_data = Data(183)

# model evaluator (linear regression)
evaluator = ModelEvaluator(LinearRegression(), r2_score)

# model evaluator (logistic regression)
# evaluator = evaluator = ModelEvaluator(Pipeline([('scaler', StandardScaler()),
#               ('model', LogisticRegression(max_iter=1000))]),
#     accuracy_score
# )

ref_point = np.array([0.05, test_data.n + 1])
ext_1, ext_2 = metrics.compute_pareto_extremes(test_data, evaluator)

#%% ===========================================================================
# 1. Single Run of My FNSGA (gives plot of pareto front)
# =============================================================================
evaluator = ModelEvaluator(
    Pipeline([('scaler', StandardScaler()),
          ('model', LogisticRegression(max_iter=1000))]),
    accuracy_score)
ext_1, ext_2 = metrics.compute_pareto_extremes(test_data, evaluator)

start = time.time()
result, fig, ax = my_FNSGA(test_data, evaluator, params = {**COMMON, **LOGREG_PARAMS}, seed=seed, plot=True)
print("Single-Run Results: My FNSGA")
print(f"Seed: {seed}")
print(f"Runtime: {time.time() - start:.1f}s")
print(f"Final PF size: {len(result)}")
 
fitness = np.array([ind.fitness for ind in result.population])
hv = HV(ref_point=ref_point)(fitness)
print(f"Hypervolume: {hv:.4f}")

if ax is None:
    fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
    edgecolors="black", label = "Nondominated Set (PF)")
ax.legend(loc='lower right', fontsize=8)
ax.set_title(f'My FNSGA\nFinal Iteration and Pareto Front | Seed {seed}')
ax.set_xlabel('Number of Features')
ax.set_ylabel('Score')
plt.show()

#%% ===========================================================================
# 2. Single Run of Tom's FNSGA
# =============================================================================
evaluator = ModelEvaluator(
    Pipeline([('scaler', StandardScaler()),
          ('model', LogisticRegression(max_iter=1000))]),
    accuracy_score)

replica_params = {**COMMON, **LOGREG_PARAMS}
 
start = time.time()
result, fig, ax = replica_FNSGA(test_data, evaluator, replica_params, seed=seed, plot=True)
print("Single Run Results: Tom's FNSGA")
print(f"Seed: {seed}")
print(f"Runtime: {time.time() - start:.1f}s")
print(f"Final ND set size: {len(result)}")
 
fitness = np.array([ind.fitness for ind in result.population])
hv = HV(ref_point=ref_point)(fitness)
print(f"Hypervolume: {hv:.4f}")

if ax is None:
    fig, ax = plt.subplots(figsize=(7, 5))
 
ax.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
    edgecolors="black", label = "Final ND Set")
ax.legend(loc='lower right', fontsize=8)
ax.set_title(f"FNSGA Replica\nFinal Iteration and Pareto Front | Seed {seed}")
ax.set_xlabel('Number of Features')
ax.set_ylabel('Score')
plt.show()

 

#%% =============================================================================
# 3. Multi-algorithm comparison run 
# =============================================================================
from algorithms import *

seed_array = np.random.choice(10000, size = 1, replace = False)
extreme_1, extreme_2 = metrics.compute_pareto_extremes(test_data, evaluator)

algorithms = {"replica": replica_FNSGA, 
              "my": my_FNSGA,
              'nsga2': make_pymoo_runner(make_nsga2),
              'spea2': make_pymoo_runner(make_spea2),
              'smsemoa':  make_pymoo_runner(make_smsemoa)}

params = {"replica": {**COMMON, **REPLICA}, 
          "my": COMMON,
          'nsga2': COMMON,
          'spea2': COMMON,
          'smsemoa': COMMON
          }

results = metrics.run_metrics(seed_array, test_data, evaluator,
                      ref_point, extreme_1, extreme_2, 
                      my_FNSGA, {**COMMON, **LINREG_PARAMS})

df = pd.DataFrame(results)

summary = (df.groupby('algorithm')
             .agg({
                 'hypervolume': ['mean', 'std'],
                 'spacing':     ['mean', 'std'],
                 'spread':      ['mean', 'std'],
                 'pf_size':     ['mean', 'std'],
                 'time':        ['mean', 'std'],
             })
             .round(4))
print(summary)

# --- Friedman test ---
print("\nFriedman test")
for metric in ['hypervolume', 'spread', 'spacing', 'time']:
    groups = [
        df[df['algorithm'] == alg][metric].values
        for alg in algorithms
    ]
    stat, p = friedmanchisquare(*groups)
    print(f"  {metric:12s}: stat={stat:.4f}, p={p:.4f}")
 
#%%

with open('results.pkl', 'wb') as f:
    pickle.dump(df, f)
print("Full results saved to results.pkl")
 


#The Wilcoxon signed-rank test tests the null hypothesis that two related 
#paired samples come from the same distribution. (from scipy.stats documentation)
from scipy.stats import wilcoxon
print("Wilcoxon signed-rank test")
for metric in ["HV", "spread", "spacing", "time"]:
    my = [r[f"{metric}_my"] for r in results]
    replica = [r[f"{metric}_replica"] for r in results]
    diff = np.subtract(my, replica)
    res = wilcoxon(diff)
    print(f"{metric}: (statistic, p-val) = ({res.statistic:.4f}, {res.pvalue:.8f})")


































