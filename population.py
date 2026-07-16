#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:43:40 2026

@author: erinfortin
"""
from individual import Individual
import numpy as np

class Population:
    def __init__(self,N):
        self.N = N
        self.population = []
    
    def initialize(self, n_features, feat_importances, seeding_prop, ones_prop ,rng):
        n_o = round(self.N * ones_prop)
        n_s = round(self.N * seeding_prop)
        
        importances_idxs = np.argsort(feat_importances)
        
        for i in range(n_s):
            k = importances_idxs[i]
            self.population.append(Individual(np.eye(n_features)[k]))
        for i in range(n_s , n_s + n_o): 
            self.population.append(Individual(np.ones(n_features)))
        for i in range(n_s + n_o, self.N):
            self.population.append(Individual(rng.integers(2, size = n_features)))

    
    @property
    def chromosomes(self):
        return np.vstack([ind.chromosome for ind in self.population])
    
    @property
    def fitness(self):
        return np.vstack([ind.fitness for ind in self.population])
    
    def evaluate(self, data, model, scorer):
        for individual in self.population:
            individual.evaluate_fitness(data, model, scorer)
    
    def __len__(self):
        return len(self.population)
        
        
        