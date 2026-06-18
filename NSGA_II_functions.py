#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:00:51 2026

@author: erinfortin
"""
import numpy as np 
from population_class import Population
from individual_class import Individual

def dominates(p,q):
    '''
    Determines if an individual p dominates another individual q
    '''
    # first condition: p <= q for all objectives
    better_or_equal = all(
        a <= b for a,b in zip(p.fitness, q.fitness)
        )
    # second condition: p < q for at least one objective
    strictly_better = any(
        a < b for a,b in zip(p.fitness, q.fitness)
        )
    
    return better_or_equal and strictly_better
    

def fast_non_dominated_sort(population):
    '''
    Performs fast, non-dominating sorting on a population as in Deb et al (2002).
        Inputs: a population of individuals
        Output: The fronts identified
    '''
    fronts = [[]]    

    for p in population.population:
        p.dominated_solutions = []
        p.domination_count = 0
        
        for q in population.population:
            if p is q:
                continue
            
            if dominates(p, q):
                p.dominated_solutions.append(q)
            
            elif dominates(q, p):
                p.domination_count += 1
        
        if p.domination_count == 0:
            p.rank = 1
            fronts[0].append(p)
    
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 2 
                    next_front.append(q)
                    
        i += 1
        fronts.append(next_front)
        
    fronts.pop()
    return fronts

def crowding_distance(front):
    '''
    Calculates the crowding distance of a front
    '''
    if len(front) == 0:
        return
    
    n_obj = len(front[0].fitness)
    
    for ind in front:
        ind.crowding_distance = 0
    
    for m in range(n_obj):
        front.sort(key=lambda x: x.fitness[m])
        front[0].crowding_distance = np.inf
        front[-1].crowding_distance = np.inf
        
        
        f_min = front[0].fitness[m]
        f_max = front[-1].fitness[m]
                
        if f_min == f_max:
            continue
         
        for i in range(1, len(front) - 1):
             front[i].crowding_distance += (
                 front[i+1].fitness[m] 
                 - front[i-1].fitness[m]
                 ) / (f_max-f_min)
        

def crowded_comparison(a, b, rng):
    '''
    Defines the crowded-comparison operator as in Deb et al (2002).
        Inputs: two individuals a and b
        Output: the preferred solution according to crowded-comparison
    '''
    if a.rank < b.rank:
        return a

    if b.rank < a.rank:
        return b

    if a.crowding_distance > b.crowding_distance:
        return a
    
    if b.crowding_distance > a.crowding_distance:
        return b

    return a if rng.random() < 0.5 else b

def tournament_selection(population, rng):
    '''
    Chooses two random individuals in a population and 
    determine the preferred solution using crowded-comparison.
    '''
    idx = rng.choice(len(population.population), 2, replace = False)
    
    a, b = population.population[idx[0]], population.population[idx[1]]

    return crowded_comparison(a, b, rng)

def k_point_crossover(parent1, parent2, p_c, rng):
    '''
    Performs single-point crossover at k-th position between two parents
    to obtain two offspring solutions
        Inputs: Individuals: parent1, parent2
                Crossover probability: p_c
        Output: Individuals: offspring1, offspring2
    '''
    
    offspring1 = parent1.chromosome.copy()
    offspring2 = parent2.chromosome.copy()
    
    if rng.random() < p_c:
        k = rng.integers(1,len(offspring1))
        
        offspring1 = np.concatenate([
            parent1.chromosome[:k],
            parent2.chromosome[k:]
        ])

        offspring2 = np.concatenate([
            parent2.chromosome[:k],
            parent1.chromosome[k:]
        ])
    return Individual(offspring1), Individual(offspring2)

def uniform_crossover(parent1, parent2, p_c, rng):
    ''' 
    Performs uniform crossover as consistent with NSGA2 and Pymoo package 
    '''
    offspring1 = parent1.chromosome.copy()
    offspring2 = parent2.chromosome.copy()
    
    if rng.random() < p_c:
        mask = rng.random(len(parent1.chromosome)) < 0.5
        
        offspring1 = np.where(
            mask,
            parent1.chromosome,
            parent2.chromosome)
        
        offspring2 = np.where(
            mask,
            parent2.chromosome,
            parent1.chromosome)
        
    return Individual(offspring1), Individual(offspring2)
        
        
    

def mutation(individual, p_m, rng):
    '''
    Performs bit-wise mutation on an individual with probability p_m.
    '''
    chromosome = individual.chromosome.copy()
    mutated = False
    for i in range(len(chromosome)):
        if rng.random() < p_m:
            chromosome[i] = 1 - chromosome[i]
            mutated= True
    individual.chromosome = chromosome
    # reset fitness of mutated chromosome
    if mutated: 
        individual.fitness = None 

def generation_algorithm(parent_pop, offspring_pop, N):
    '''
    Creates the t-th generation of the algorithm. 
        Inputs: Populations: parent_pop, offspring_pop
                Population size: N
        Outputs: Next generation population 
    '''
    
    combined = Population(N)
    combined.population = (parent_pop.population + offspring_pop.population)
    
    # remove duplicate chromosomes
    unique = {}
    for ind in combined.population:
        key = tuple(ind.chromosome)
        unique[key] = ind

    combined.population = list(unique.values())
    
    
    fronts = fast_non_dominated_sort(combined)
    
    new_population = []
    for front in fronts:
        crowding_distance(front)
        
        if len(new_population) + len(front) <= N:
            new_population.extend(front)
        else:
            front.sort(key=lambda x: x.crowding_distance, reverse=True)
            
            remaining = N - len(new_population)
            
            new_population.extend(front[:remaining])
            break
    next_population = Population(N)
    next_population.population = new_population
    return next_population








    