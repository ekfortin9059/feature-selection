#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:29:35 2026

@author: erinfortin
"""

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None

        self.rank = None
        self.crowding_distance = 0
        
        self.domination_count = 0
        self.dominated_solutions = []
        
    def evaluate(self, data, evaluator):
        if self.fitness is not None:
            return
        self.fitness = evaluator.fitness(data, self.chromosome)