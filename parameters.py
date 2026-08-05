#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:48:31 2026

@author: erinfortin
"""

import pandas as pd

my_params_df = pd.read_csv('tuned_parameters.csv', index_col=0)

replica_params_df = pd.read_csv('replica_parameters.csv', index_col=0)

MY_PARAMS = my_params_df.to_dict(orient='index')
REPLICA_PARAMS = replica_params_df.to_dict(orient='index')

def get_algorithm_params(alg_name, model_name):
    if alg_name == 'replica':
        return REPLICA_PARAMS[model_name]
    else:
        # Use tuned parameters for my_FNSGA and all pymoo benchmark algorithms
        return MY_PARAMS[model_name]