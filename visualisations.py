#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 08:04:05 2026

@author: erinfortin

Initial visualisations of results
"""

import pickle
import os
import pandas as pd
import numpy as np 

results = pd.read_csv("results_final.csv")

results.info()
# some data is missing and others have -inf values. will need to rerun certain 
# datasets, and possibly change the smallest one. for now, we will get initial visualisations and stat test results with available data.

# remove rows where f1_min is -inf (corrupted fronts - sum(num features incl = 0), so score treated as -inf)
df_clean = results[results['f1_min'] != -np.inf].copy()

# remove rows where spread/spacing are NaN (single-point fronts - smallest dataset produced single-feature fronts i.e. one nondominated sol (no comparison exists for extreme poionts of PF))  
df_clean = df_clean.dropna(subset=['spread', 'spacing'])

# summary of rows removed 
print(f"Original: {len(results)} rows")
print(f"Clean: {len(df_clean)} rows")
print(f"Removed: {len(results)-len(df_clean)} rows")
print("\nRemaining coverage:")
print(df_clean.groupby(['dataset','model','algorithm']).size().reset_index(name='count').to_string())


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df_clean, x="task", y="hypervolume", hue="algorithm")
plt.show()

sns.boxplot(data=df_clean, x="task", y="spread", hue="algorithm")
plt.show()

sns.boxplot(data=df_clean,x="task",y="spacing",hue="algorithm")
plt.show()

sns.boxplot(data=df_clean,x="task",y="spacing",hue="algorithm",showfliers=False)
plt.show()

sns.boxplot(data=df_clean,x="task",y="time",hue="algorithm")
plt.show()

sns.boxplot(data=df_clean,x="task",y="time",hue="algorithm",showfliers=False)
plt.show()

sns.boxplot(data=df_clean,x="task",y="pf_size",hue="algorithm")
plt.show()


spambase_ds = results[results['dataset'] == "spambase"]