#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:43:40 2026

@author: erinfortin
"""
from individual_class import Individual
import numpy as np

class Population:
    def __init__(self,N):
        self.N = N
        self.population = []

    
    def initialize(self, n_features, rng):
        self.population = [
            Individual(rng.integers(2, size = n_features))
            for _ in range(self.N)
            ]
    
    def evaluate(self, data):
        for individual in self.population:
            individual.evaluate_fitness(data)
    
    def __len__(self):
        return len(self.population)