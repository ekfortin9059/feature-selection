#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:43:40 2026

@author: erinfortin
"""
from individual import Individual
import numpy as np

class Population:
    def __init__(self):
        self.population = []
        
    def __len__(self):
        return len(self.population)
    
    @property
    def fitness(self):
        return np.vstack([ind.fitness for ind in self.population])        
    
    @property
    def chromosomes(self):
        return np.vstack([ind.chromosome for ind in self.population])
        
    def initialise(self, n_features, N, feat_importances, seeding_prop, ones_prop ,rng):
        n_o = round(N * ones_prop)
        n_s = round(N * seeding_prop)
        
        # use np.abs: strong negative coefs are just as important 
        importances_idxs = np.argsort(np.abs(feat_importances))[::-1] 
        
        # first n_s are e^k for most important feature idxs k
        for i in range(n_s): 
            k = importances_idxs[i]
            self.population.append(Individual(np.eye(n_features)[k]))
        
        # next n_o are all ones vectors: we can initialise more than one since 
        # they will be explored upon then duplicates removed after the iteration
        for i in range(n_s , n_s + n_o): 
            self.population.append(Individual(np.ones(n_features)))
        
        # remaining chromosomes are randomly generated. we use the seen set to 
        # prevent duplicate random vectors from entering. 
        seen = set()
        while len(self) < N:
            chrom = tuple(rng.integers(2, size = n_features))
            if chrom not in seen:
                seen.add(chrom)
                self.population.append(Individual(np.array(chrom)))
        
    def evaluate(self, data, model, scorer):
        for individual in self.population:
            individual.evaluate_fitness(data, model, scorer)
    
    def update_pop(self, offspring, N, n_features, rng):
        '''
        Creates the next generation by combining parent and offspring populations,
        removing duplicates (supplementing with random individuals if needed), 
        then selecting the best N individuals by rank and crowding distance. 
        '''
        import nsga_functions as nsga
            
        combined = self.population + offspring.population
        
        # use dictionary keys to check for duplicates while keeping class properties of individuals 
        unique = {}
        for ind in combined:
            unique.setdefault(tuple(ind.chromosome), ind)

        self.population = list(unique.values())

        # check if population size is large enough otherwise create random individuals 
        # (make sure they are unseen first)
        seen = set()
        while len(self) < N:
            chrom = tuple(rng.integers(2, size = n_features))
            if chrom not in seen:
                seen.add(chrom)
                self.population.append(Individual(np.array(chrom)))
            
        # create the next generation by adding the best solutions from the 
        # combined populations
        fronts = nsga.fast_non_dominated_sort(self)
        next_pop = []
        for front in fronts:
            nsga.crowding_distance(front)
            if len(next_pop) + len(front) <= N:
                next_pop.extend(front)
            else:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = N - len(next_pop)
                next_pop.extend(front[:remaining])
                break
            
        self.population = next_pop
        