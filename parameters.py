#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:48:31 2026

@author: erinfortin
"""

COMMON = {
    "population_size": 100,
    "generations": 20,
    }

SVC_PARAMS = {
    "tournament_param": 0.72,
    "crossover_prob": 0.68,
    "mutation_prob": 0.01,
    "exploration_param": 0.41,
    "ls_param": 0.61,
    "seeding_prop":0.02 ,
    "ones_prop": 0.04,
    "L": 11
}

DT_PARAMS = {
    "tournament_param": 0.63,
    "crossover_prob": 0.68,
    "mutation_prob": 0.02,
    "exploration_param": 0.53,
    "ls_param": 0.81,
    "seeding_prop":0.02 ,
    "ones_prop": 0.05,
    "L": 8
}

LOGREG_PARAMS = {
    "tournament_param": 0.61,
    "crossover_prob": 0.7,
    "mutation_prob": 0.01,
    "exploration_param": 0.55,
    "ls_param": 0.71,
    "seeding_prop":0.02 ,
    "ones_prop": 0.04,
    "L": 11
}

LINREG_PARAMS = {
    "tournament_param": 0.59,
    "crossover_prob": 0.70,
    "mutation_prob": 0.02,
    "exploration_param": 0.63,
    "ls_param": 0.43,
    "seeding_prop":0.03 ,
    "ones_prop": 0.03,
    "L": 7
}