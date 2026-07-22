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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pymoo.indicators.hv import HV
from scipy.stats import wilcoxon

from algorithms import my_FNSGA, replica_FNSGA
from population import Population
from individual import Individual
from data_class import Data
from parameters import COMMON, REPLICA
from ML_code import eval_model
import benchmark_metrics as bm

seed = np.random.choice(10000)
data = Data(COMMON['dataset_id'])

model = LinearRegression()
scorer = r2_score

#%% ===========================================================================
# 1. Single Run of My FNSGA
# =============================================================================

start = time.time()
result, fig, ax = my_FNSGA(data, model, scorer, params = COMMON, seed=seed, plot=True)
print("Single-Run Results: My FNSGA")
print(f"\nSeed: {seed}")
print(f"Runtime: {time.time() - start:.1f}s")
print(f"Final PF size: {len(result)}")
 
ref_point = np.array([1.1, data.n + 1])
fitness = np.array([ind.fitness for ind in result.population])
hv = HV(ref_point=ref_point)(fitness)
print(f"Hypervolume: {hv:.4f}")
 
ax.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
    edgecolors="black", label = "Nondominated Set (PF)")
ax.legend(loc='lower right', fontsize=8)
ax.set_title(f'My FNSGA\nFinal Iteration and Pareto Front | Seed {seed}')
plt.show()

#%% ===========================================================================
# 2. Single Run of Tom's FNSGA
# =============================================================================
replica_params = {**COMMON, **REPLICA}
 
start = time.time()
result, fig, ax = replica_FNSGA(data, model, scorer, replica_params, seed=seed, plot=True)
print("Single Run Results: Tom's FNSGA")
print(f"\nSeed: {seed}")
print(f"Runtime: {time.time() - start:.1f}s")
print(f"Final ND set size: {len(result)}")
 
ref_point = np.array([1.1, data.n + 1])
fitness = np.array([ind.fitness for ind in result.population])
hv = HV(ref_point=ref_point)(fitness)
print(f"Hypervolume: {hv:.4f}")
 
ax.scatter(fitness[:, 1], -fitness[:, 0],facecolors="none",
    edgecolors="black", label = "Final ND Set")
ax.legend(loc='lower right', fontsize=8)
ax.set_title(f"FNSGA Replica\nFinal Iteration and Pareto Front | Seed {seed}")
plt.show()

#%% =============================================================================
# 3. Multi-seed run (Wilcoxon signed rank)
# =============================================================================
ref_point = np.array([1.1, data.n + 1])

seed_array = np.random.integers(10000, size = 20, replace = False)
extreme_1, extreme_2 = bm.compute_pareto_extremes( data, eval_model, model, scorer)

algorithms = {"replica": replica_FNSGA, "my": my_FNSGA}
params = {"replica": {**COMMON, **REPLICA}, "my": COMMON}

results = bm.run_FNSGA_metrics(seed_array, data, model, scorer,
                      ref_point, extreme_1, extreme_2, 
                      algorithms, params)

# change to mean (sd) table with wilcoxon results
import pandas as pd
table = {}
for r in results:
    seeds = f"Seed {r['seed']}     "
    table[seeds] = {
                    "": "my FNSGA  rep FNSGA |",
                    "Hypervolume": f"{r['HV_my']:.2f}    {r['HV_replica']:.2f} |",
                    "Spread": f"{r['spread_my']:.2f}      {r['spread_replica']:.2f} |",
                    "Spacing": f"{r['spacing_my']:.2f}      {r['spacing_replica']:.2f} |",
                    "Runtime (s)": f"{r['time_my']:.2f}     {r['time_replica']:.2f} |"}
    
    df = pd.DataFrame(table)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(df)
    

print("Wilcoxon signed-rank test")
wilcoxon_res = {"HV": [], "spread": [], "spacing": [], "time": []}
for metric in ["HV", "spread", "spacing", "time"]:
    my = [r[f"{metric}_my"] for r in results]
    replica = [r[f"{metric}_replica"] for r in results]
    diff = np.subtract(my, replica)
    res = wilcoxon(diff)
    print(f"{metric}: (statistic, p-val) = ({res.statistic:.4f}, {res.pvalue:.8f})")
    


