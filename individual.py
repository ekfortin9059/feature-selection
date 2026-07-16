#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:29:35 2026

@author: erinfortin
"""
from ML_code import eval_model
import numpy as np

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None

        self.rank = None
        self.crowding_distance = 0
        
        self.domination_count = 0
        self.dominated_solutions = []
        
    def evaluate_fitness(self, data, model, scorer):
        # no need to re-evaluate if already done
        if self.fitness is not None:
            return
        # use -obj1 to convert to minimisation
        self.fitness = [
            -eval_model(data, self.chromosome, model, scorer),
            np.sum(self.chromosome)
            ]

