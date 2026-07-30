#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
using computational results, complete statistical testing between algorithms using
friedman test and (if significant) pairwise wilcoxon tests

@author: erinfortin
"""

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def run_statistics(results, metric, baseline="my"):
    '''
    compare all other algorithms to mine using friedman test, then if sig 
    result,apply pairwise wilcoxon test and further the holm correction 
    for multiple comparisons
    '''

    stats = []

    # analyse each dataset/model separately
    for (dataset, model), group in results.groupby(["dataset", "model"]):

        pivot = group.pivot(
            index="seed",
            columns="algorithm",
            values=metric
        )

        # remove seeds with missing values
        pivot = pivot.dropna()

        # need at least two algorithms
        if pivot.shape[1] < 2:
            continue

        algs = list(pivot.columns)

        # Friedman test
        stat, p = friedmanchisquare(*[pivot[a] for a in algs])

        stats.append({
            "dataset": dataset,
            "model": model,
            "metric": metric,
            "test": "Friedman",
            "algorithm": None,
            "statistic": stat,
            "pvalue": p,
            "corrected_p": None,
            "significant": p < 0.05
        })

        # Post-hoc tests
        if p < 0.05 and baseline in algs:

            comparisons = []
            pvals = []

            for alg in algs:

                if alg == baseline:
                    continue
                
                # wilcoxon test 
                stat_w, p_w = wilcoxon(pivot[baseline],pivot[alg])

                comparisons.append((alg, stat_w))
                pvals.append(p_w)

            # Holm correction: multiple comparisons (seeds) often give at least one false positive (0.05 sig level)
            # holm-bonferroni correction controls the family-wise error rate
            reject, p_corr, _, _ = multipletests(
                pvals,
                alpha=0.05,
                method="holm"
            )

            for (alg, stat_w), raw_p, corr_p, sig in zip(comparisons, pvals, p_corr, reject):

                stats.append({
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "test": "Wilcoxon",
                    "algorithm": alg,
                    "statistic": stat_w,
                    "pvalue": raw_p,
                    "corrected_p": corr_p,
                    "significant": sig
                })

    return pd.DataFrame(stats)


metrics = ["hypervolume","spread","spacing","time","pf_size"]

results = pd.read_csv("results_final.csv")

output = {}

for metric in metrics:
    stats = run_statistics(results, metric)
    output[metric] = stats
    stats.to_csv(f"statistics/{metric}_stats.csv", index=False)


