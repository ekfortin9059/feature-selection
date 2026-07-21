#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:00:51 2026

@author: erinfortin
"""
import numpy as np 
from population import Population
from individual import Individual

# =============================================================================
# Helper Functions
# =============================================================================

def update_archive(pop_t, archive, N_archive):
    '''
    Updates the archive by combining the current population and the existing
    archive, then keeping the best N_archive individuals by rank and crowding.
    '''
    pool = Population()
    pool.population = [*pop_t.population, *archive.population] # pop_t.population + archive.population

    unique = {}

    for ind in [*pop_t.population, *archive.population]:
        key = tuple(np.asarray(ind.chromosome, dtype=int))
        unique.setdefault(key, ind)
    
    pool.population = list(unique.values())

    archive_temp = []
    fronts = fast_non_dominated_sort(pool)

    for front in fronts:
        crowding_distance(front)

        if len(archive_temp) + len(front) <= N_archive:
            archive_temp.extend(front)
        else:
            front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            remaining = N_archive - len(archive_temp)
            archive_temp.extend(front[:remaining])
            break

    archive.population = archive_temp
    return archive

def extend_archive(pop_t, archive):
    '''
    Pure union of current population into archive — no truncation.
    Deduplicates by chromosome to avoid redundant eval_model calls later.
    '''
    seen = {tuple(ind.chromosome) for ind in archive.population}
    for ind in pop_t.population:
        if tuple(ind.chromosome) not in seen:
            archive.population.append(ind)
            seen.add(tuple(ind.chromosome))
    return archive


# =============================================================================
# Dominance-Related Functions
# =============================================================================
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
    Sets rank on each individual and returns list of fronts.
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
    Calculates the crowding distance of a front as in Deb et al (2002).
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
    Prefers lower rank; breaks ties by higher crowding distance.

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


# =============================================================================
# Operator Functions
# =============================================================================
def k_point_crossover(parent1, parent2, rng):
    '''
    Performs single-point crossover at a random position k between two parents.
    Returns two offspring individuals.
    Consistent with Algorithm 3 in the FNSGA paper.

    '''
    k = rng.integers(1, len(parent1.chromosome))
    
    offspring1 = np.concatenate([
        parent1.chromosome[:k],
        parent2.chromosome[k:]
    ])

    offspring2 = np.concatenate([
        parent2.chromosome[:k],
        parent1.chromosome[k:]
    ])
    return Individual(offspring1), Individual(offspring2)

def uniform_crossover(parent1, parent2, crossover_prob, rng):
    ''' 
    Performs uniform crossover consistent with NSGA-II and pymoo.
    Each bit is independently swapped with probability 0.5.
    '''

    offspring1 = parent1.chromosome.copy()
    offspring2 = parent2.chromosome.copy()
    
    if rng.random() < crossover_prob:
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
    Performs bit-wise mutation: each bit is independently flipped
    with probability p_m. 
    '''
    chromosome = individual.chromosome.copy()
    flip_mask = rng.random(len(chromosome)) < p_m
    if flip_mask.any():
        chromosome[flip_mask] = 1 - chromosome[flip_mask]
        individual.chromosome = chromosome
        individual.fitness = None

# =============================================================================
# NSGA-II - Specific Functions
# =============================================================================
def generation_algorithm(parent_pop, offspring_pop, N):
    '''
    Creates the next generation by combining parent and offspring populations,
    removing duplicates, then selecting the best N individuals by rank and
    crowding distance. Used in the standard NSGA-II loop.
    '''
    
    ##### replace duplicates with random individuals
    
    combined = Population()
    combined.population = (parent_pop.population + offspring_pop.population)
    combined.population = (parent_pop.population + offspring_pop.population)
    
    # check if population size is large enough otherwise create random individuals 
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
        
    next_population = Population()
    next_population.population = new_population
    return next_population

# =============================================================================
# Selection Functions
# =============================================================================
def tournament_selection(population, rng):
    '''
    Standard binary tournament selection using crowded comparison.
    '''
    idx = rng.choice(len(population.population), 2, replace = False)
    
    a, b = population.population[idx[0]], population.population[idx[1]]

    return crowded_comparison(a, b, rng)



def special_tournament_select(population, tournament_param, 
                              exploration_param, N, rng):
    '''
    Specialised tournament selection as in Algorithm 4 of the FNSGA paper.
    Selects individuals via two criteria:
        - a individuals chosen by non-dominated rank (lower is better)
        - b individuals chosen by crowding distance (higher is better)
    The exploration_param epsilon controls the balance between the two.
    
    NOTE: ranks and crowding distances must already be set on the population
    before calling this function (no internal re-sort).
    '''

    a = round(exploration_param * tournament_param * N) # num individuals to be selected from nondominated ranks
    b = round((1-exploration_param) * tournament_param * N) # num individuals to be selected from crowding ranks
    
    ranks = np.argsort([population.population[i].rank for i in range(N)])
    crowdings = np.argsort([population.population[i].crowding_distance for i in range(N)])
    
    children = Population()
    # select a individuals by rank  (lower rank = better)
    for _ in range(a):
        j = rng.integers(0,len(ranks))
        k = rng.integers(0,len(ranks))
        children.population.append(population.population[ranks[min(j,k)]])
    
    # select b individuals by crowding distance (higher is better)
    for _ in range(b):
        j = rng.integers(0,len(crowdings))
        k = rng.integers(0,len(crowdings))
        children.population.append(population.population[crowdings[max(j,k)]])
    
    return children

def evolutionary_selection(population, crossover_prob, mutation_prob, 
                           tournament_param, exploration_param, N, N_return, rng):
    '''
    Evolutionary Selection operator as in Algorithm 3 of the FNSGA paper.
    Selects a mating pool C via specialised tournament selection, then
    generates N_return unique offspring via crossover and/or mutation.
    
    Crossover (k-point) applied if r < pc.
    Mutation applied if pc <= r < pc + pm.
    Otherwise individual is cloned.
    
    Returns a Population of N_return unique, non-empty offspring.
    '''

    C = special_tournament_select(population, tournament_param, 
                                  exploration_param, N, rng)
    selected = []
    seen = set()
    
    while len(selected) < N_return:
        r = rng.uniform(0,1)
        
        if r < crossover_prob:
            idx = rng.choice(len(C), 2, replace = False)
            c1, c2 = C.population[idx[0]], C.population[idx[1]] 
            o1, o2 = k_point_crossover(c1, c2, rng)    
            
            for o in [o1, o2]:
                chrom = tuple(o.chromosome.copy())
                if chrom not in seen and len(selected) < N_return:
                    if np.sum(o.chromosome) > 0:        
                        seen.add(chrom)
                        selected.append(chrom)
        else:
            c = C.population[rng.choice(len(C))]
            mutant = Individual(c.chromosome.copy())
            if r < crossover_prob + mutation_prob:
                mutation(mutant, mutation_prob, rng)
            
            chrom = tuple(mutant.chromosome.copy())
            if chrom not in seen and len(selected) < N_return:
                if np.sum(mutant.chromosome) > 0:       
                    seen.add(chrom)
                    selected.append(chrom)
    
    pop = Population()
    pop.population = [Individual(np.array(c)) for c in selected]
    
    return pop

# =============================================================================
# Local Search Operators 
# =============================================================================    

### exact replica as paper for comparison
def original_local_search(population, ls_param, feat_scores, N_return, rng):
    '''
    Local Search operator as described in Algorithm 5 of the FNSGA paper.
    Returns the first N_return unique valid candidates.
    '''
    # select first ls_param nondominated individuals
    nd_ranks_idxs = np.argsort([ind.rank for ind in population.population])[:ls_param]
    C_l = [population.population[i].chromosome.copy() for i in nd_ranks_idxs]
    
    # sort feature indices by score
    sorted_scores = np.argsort(feat_scores)
    
    # partition sorted features in to 4 subsets 
    feat_partition = np.array_split(sorted_scores, 4)
    
    selected = []
    seen = set()
    
    for subset in feat_partition:
        if len(subset) < 2:
            continue
        for c in C_l:
            n_feat = len(c)
            for _ in range(n_feat // 2):
                c_new = c.copy()
                x1, x2 = rng.choice(subset, 2, replace=False)
                j = max(x1, x2)
                c_new[j] = 1 - c_new[j]
                chrom = tuple(c_new)
                if chrom not in seen and c_new.sum() > 0:
                    seen.add(chrom)
                    selected.append(c_new)
    
    pop = Population()
    pop.population = [Individual(np.array(c)) for c in selected[:N_return]]
    return pop                        
        

# Add: If a feature has a high score, we want the likelihood of including that feature to be high
def add_local_search(population, feat_scores, N_return, rng):
    '''
    Add Local Search: for each solution, probabilistically turns features ON.
    Features with high score relative to the individual's current active features
    are more likely to be included.
    p(add feature f) = feat_scores[f] / sum(active feature scores)
    Generates N_return unique valid candidates.
    '''

    selected = []
    seen = set()
    inds = population.population
    
    for ind in inds:
        c_new = ind.chromosome.copy()
        
        total_score = c_new @ feat_scores
        p = feat_scores / total_score
        
        rands = rng.random(len(c_new))
        c_new = np.where(rands < p, 1, c_new)
    
    
        chrom = tuple(c_new)
        if chrom not in seen and c_new.sum() > 0:
            seen.add(chrom)
            selected.append(c_new)
        
        if len(selected) >= N_return:
            break
    
    pop = Population()
    pop.population = [Individual(np.array(c)) for c in selected]
    return pop


# Remove: If a feature has a low score, we want the likelihood of including that feature to be low
def remove_local_search(population, feat_scores, N_return, rng):
    '''
    Remove Local Search: for each solution, probabilistically turns features OFF.
    Features with low score relative to the individual's active feature scores
    are more likely to be removed.
    p(remove feature f) = 1 - feat_scores[f] / sum(active feature scores)
    Equivalently: remove if rand > feat_scores[f] / total_score
    Generates N_return unique valid candidates.
    '''

    selected = []
    seen = set()
    inds = population.population
    
    for ind in inds:
        c_new = ind.chromosome.copy()
        
        total_score = c_new @ feat_scores
        p = feat_scores / total_score
        
        rands = rng.random(len(c_new))
        c_new = np.where(rands > p, 0, c_new)
    
    
        chrom = tuple(c_new)
        if chrom not in seen and c_new.sum() > 0:
            seen.add(chrom)
            selected.append(c_new)
        
        if len(selected) >= N_return:
            break
    
    pop = Population()
    pop.population = [Individual(np.array(c)) for c in selected]
    return pop


# Merge: take pairs of ND solutions and merge the solution (feat = 1 if either solution includes it)
def merge_local_search(population, N_return, rng):
    '''
    Merge Local Search: randomly picks pairs of non-dominated solutions and
    merges them via OR (feature included if present in either solution).
    Generates N_return unique valid candidates.
    '''

    selected = []
    seen = set()
    inds = population.population

    
    for i in range(len(inds)):
        for j in range(i+1, len(inds)):
            c_new = np.maximum(inds[i].chromosome.copy(), inds[j].chromosome.copy())
            
            chrom = tuple(c_new)
            if chrom not in seen and c_new.sum() > 0: 
                seen.add(tuple(c_new))
                selected.append(c_new)
        if len(selected) >= N_return:
            break
    selected = selected[:N_return]
    pop = Population()
    pop.population = [Individual(np.array(c)) for c in selected]
    return pop

# Add/Remove: high feature importance likely to be added 
# low feat importance likely to be removed
def add_remove_local_search(population, feat_scores, N_return, rng):
    '''
    Add-Remove Local Search: combines add and remove in a single pass.
    For each feature:
        - if currently 0 and rand < p: turn ON  (add high-importance feature)
        - if currently 1 and rand > p: turn OFF (remove low-importance feature)
    where p = feat_scores[f] / sum(active feature scores).
    Generates N_return unique valid candidates.
    '''

    selected = []
    seen = set()
    inds = population.population

    for ind in inds:
        c_new = ind.chromosome.copy()
        total_score = c_new @ feat_scores
        if total_score == 0:
            continue
        
        p = feat_scores / total_score
        rands = rng.random(len(c_new))

        add_mask = (c_new == 0) & (rands < p)
        remove_mask = (c_new == 1) & (rands > p)
        c_new[add_mask] = 1
        c_new[remove_mask] = 0

        chrom = tuple(c_new)
        if chrom not in seen and c_new.sum() > 0:
            seen.add(chrom)
            selected.append(c_new)
        if len(selected) >= N_return:
            break
    
    pop = Population()
    pop.population = [Individual(np.array(c)) for c in selected]
    return pop
            
    
    