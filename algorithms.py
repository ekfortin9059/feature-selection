#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection Algorithms

Sections
0. Preliminaries
1. Improved FNSGA 
2. Replica of FNSGA as in Murarik & Searle (2025)
3. Pymoo Algorithm Setup
"""
# =============================================================================
# 0. Preliminaries
# =============================================================================
import numpy as np 
import matplotlib.pyplot as plt

from population import Population
import operators

# =============================================================================
# 1. Improved FNSGA
# =============================================================================

def my_FNSGA(data, model_evaluator, params, seed=None, plot=False):
    '''
    Improvement of the Feature Nondominated Sorting Genetic Algorithm from 
    Murarik & Searle (2025). 
    
    Takes as inputs: data, a ML model and scoring metric, algorithm parameters,
    a seed for reproducibility, and plotting option. 
    plot = True provides iteration-level plots demonstrating where solutions 
    come from (Evo Select or a Local Search)
    
    Returns the pareto front after T generations 
    '''
    rng = np.random.default_rng(seed)
    
    # Extract parameters
    N = int(params["population_size"])
    T = int(params["generations"])

    p_c = params["crossover_prob"]
    p_m = params["mutation_prob"]
    seeding_prop = params["seeding_prop"]
    ones_prop = params["ones_prop"]
    ls_start = params["ls_start"]
    ls_end = params["ls_end"]
    
    # feature importances from ML model
    feat_importances = model_evaluator.feature_importances(data)
    
    # generate initial population (including rank and crowding distance)
    pop_t = Population()
    pop_t.initialise(data.n, N, feat_importances, seeding_prop, ones_prop ,rng)
    pop_t.evaluate(data, model_evaluator)
    
    fronts = operators.fast_non_dominated_sort(pop_t)
    for front in fronts:
        operators.crowding_distance(front)
     
    # local search geometric heating rate
    growth_rate = (ls_end / ls_start) ** (1 / (T - 1)) if T > 1 else 1.0
    
    # history for plotting iterations
    last_fig = None
    last_ax = None
    if plot: 
        history = []
    
    for t in range(T):
        # get normalised feature scores of nondominated solutions (for local search)
        A = pop_t.chromosomes
        model_scores = pop_t.fitness[:, 0]
        denom = A.sum(axis=0)
        denom[denom == 0] = 1
        feat_scores = (A.T @ -model_scores) / denom
        
        # extract non-dominated set from population
        nd_set = Population()
        nd_set.population = [ind for ind in pop_t.population if ind.rank == 1]
        
        ls_t = min(ls_start * (growth_rate ** t), ls_end)
        
        # amount of individuals for offspring coming from LS and ES
        N_local = round(N * ls_t)
        N_evo = N - N_local
        
        # Evolutionary Selection
        evo_pop = operators.evolutionary_selection2(pop_t, p_c, p_m, N_evo, rng)
        evo_pop.evaluate(data, model_evaluator)
        
            
        LS_return = min(len(nd_set), N_local // 4)

        add = operators.add_local_search(nd_set, feat_scores, LS_return, rng)
        remove = operators.remove_local_search(nd_set, feat_scores, LS_return, rng)
        addremove = operators.add_remove_local_search(nd_set, feat_scores, LS_return, rng)
        merge = operators.merge_local_search(nd_set, round(N_local - LS_return*3), rng)

        ls_pop = Population()
        ls_pop.population = [*add.population, *remove.population, *addremove.population, *merge.population] 
        ls_pop.evaluate(data, model_evaluator)
            
        if plot:
            def _get_fitness(pop):
                f = [ind.fitness for ind in pop.population]
                return np.array(f) 
            history.append({
                'iteration': t + 1,
                'evo': _get_fitness(evo_pop),
                'add': _get_fitness(add),
                'remove': _get_fitness(remove),
                'addremove': _get_fitness(addremove),
                'merge': _get_fitness(merge),
            })

        # next generation
        offspring = Population()
        offspring.population = [*evo_pop.population,*ls_pop.population]        
        rng.shuffle(offspring.population)
         
        # update to next population
        pop_t.update_pop(offspring, N, data.n, rng)
        

    # final ND solution set 
    fronts = operators.fast_non_dominated_sort(pop_t)
    final_nd = Population()
    final_nd.population = fronts[0]
    
    if plot:
        for h in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            for key, label, color, marker in [
                ('evo', 'New solutions: Evolutionary Select', 'red',  'o'),
                ('add', 'New solutions: Local Search: Add',   'orange',   '*'),
                ('remove', 'New solutions: Local Search: Remove',   'green',   '^'),
                ('merge', 'New solutions: Local Search: Merge',   'blue',   's'),
                ('addremove', 'New solutions: Local Search: Add-Remove',   'purple', 'd')

            ]:
                d = h[key]
                if len(d) == 0:
                    continue
                ax.scatter(d[:, 1], -d[:, 0], label=label, alpha=0.6,
                           color=color, marker=marker, s=30)
            ax.set_xlabel('Number of Features')
            ax.set_ylabel('R²')
            ax.set_title(f'My FNSGA\nIteration {h["iteration"]} | Seed {seed}')
            ax.legend(loc='lower right', fontsize=8)
            plt.tight_layout()
            if h["iteration"] == T:
                last_fig = fig
                last_ax = ax
            else:
                plt.show()

    return final_nd, last_fig, last_ax

# =============================================================================
# 2. Replica of FNSGA 
# =============================================================================

def replica_FNSGA(data, model_evaluator, params, seed=None, plot=False):
    '''
    Faithful replication of the Feature Nondominated Sorting Algorithm as found 
    in Algorithm 1 of the paper by Muarik & Searle (2025). 
    '''
    rng = np.random.default_rng(seed)

    N = int(params["population_size"])
    T = int(params["generations"])

    p_c = params["crossover_prob"]
    p_m = params["mutation_prob"]

    tournament_param = params["tournament_param"]
    exploration_param = params["exploration_param"]

    seeding_prop = params["seeding_prop"]
    ones_prop = params["ones_prop"]
    ls_param = params["ls_param"]
    
    L = int(params["L"])
    k = int(params["k"])
    
    
    # get feature importances based on model evaluation (using selectfrommodel as in paper)
    s0 = model_evaluator.feature_importances_sfm(data)
    
    # initialise population and evaluate
    pop_t = Population()
    pop_t.initialise(data.n, N, s0, seeding_prop, ones_prop, rng)
    
    # Evaluate model scores, store in archive
    pop_t.evaluate(data, model_evaluator)
    archive = Population()
    archive = operators.extend_archive(pop_t, archive)
    
    # get prop of individuals coming from each operator
    N_l = round(N * ls_param)
    N_e = N - N_l

    last_fig = None
    last_ax = None
    history = [] if plot else None

    for t in range(T):
        # extract individuals and scores from archive
        A = archive.chromosomes
        model_scores = archive.fitness[:,0] # -R^2
        
        # update feature scores
        denom = A.sum(axis = 0)
        denom[denom==0] = 1
        feat_scores = A.T @ -model_scores
        
        # ensure ranks and crowding distances are set on archive
        fronts = operators.fast_non_dominated_sort(archive)
        for front in fronts:
            operators.crowding_distance(front)
    
        # get nondominated set
        nd_set = Population()
        nd_set.population = [ind for ind in archive.population if ind.rank == 1]
        
        # evolutionary selection
        evo_pop = operators.evolutionary_selection(archive, p_c, p_m, 
                                                   tournament_param, 
                                                   exploration_param,
                                                   len(archive), N_e, rng, k = k)

        # local search
        ls_pop = operators.original_local_search(nd_set, min(L, len(nd_set)), 
                                                 feat_scores, N_l, rng)
        
        # shuffle
        offspring = Population()
        offspring.population = [*evo_pop.population, *ls_pop.population]
        rng.shuffle(offspring.population)
        offspring.evaluate(data, model_evaluator)

        # update main archive with this iteration's solutions
        archive = operators.extend_archive(offspring, archive)
        pop_t = offspring
                
        if plot:
            def _get_fitness(pop):
                f = [ind.fitness for ind in pop.population if ind.fitness is not None]
                return np.array(f) if f else np.empty((0, 2))
            history.append({
                'iteration': t + 1,
                'evo': _get_fitness(evo_pop),
                'ls': _get_fitness(ls_pop)
            })
        
    # get nondominated set to return 
    fronts = operators.fast_non_dominated_sort(archive)
    final_nd = Population()
    final_nd.population = fronts[0]

    if plot:
        for h in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            for key, label, color, marker in [
                ('evo', 'New solutions: Evolutionary Select', 'red',  'o'),
                ('ls', 'New solutions: Local Search',        'green',   '*'),

            ]:
                d = h[key]
                if len(d) == 0:
                    continue
                ax.scatter(d[:, 1], -d[:, 0], label=label, alpha=0.6,
                           color=color, marker=marker, s=30)
            ax.set_xlabel('Number of Features')
            ax.set_ylabel('R²')
            ax.set_title(f'FNSGA replica\nIteration {h["iteration"]} | Seed {seed}')
            ax.legend(loc='lower right', fontsize=8)
            plt.tight_layout()
            if h["iteration"] == T:
                last_fig = fig
                last_ax = ax
            else:
                plt.show()

    # return final ND set and final iteration plot
    return final_nd, last_fig, last_ax

# ===========================================================================
# 3. Pymoo algorithm setup
# =============================================================================
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.nsde import NSDE
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.gde3 import GDE3
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling



#from pymoo.algorithms.moo.age2 import AGEMOEA2



class FeatureSelectionProblem(ElementwiseProblem):
    def __init__(self, data, evaluator):
        super().__init__(
            n_var=data.n,
            n_obj=2,
            xl=0,
            xu=1,
            vtype=bool
        )
        self.data = data
        self.evaluator = evaluator
 
    def _evaluate(self, X, out, *args, **kwargs):
        X_bool = X.astype(bool)
        # if no features are selected, make sure obj values are always dominated
        if not X_bool.any():
            out['F'] = np.array([1.1, 0])  
            return
        score = self.evaluator.score(self.data, X_bool)
        out['F'] = np.array([-score, X_bool.sum()])
 

def make_pymoo_runner(alg_factory):
    from individual import Individual
 
    def run(data, evaluator, params, seed=None, plot=False):
        pymoo_seed = np.random.RandomState(seed) if seed is not None else None
        
        problem = FeatureSelectionProblem(data, evaluator)
        algorithm = alg_factory(params, seed)
        res = minimize(
            problem, algorithm,
            termination=('n_gen', params['generations']),
            seed=pymoo_seed, verbose=False
        )
        nd = type('Population', (), {'population': [], 'fitness': None})()
        inds = []
        for i, x in enumerate(res.X):
            ind = Individual(x)
            ind.fitness = list(res.F[i])
            inds.append(ind)
        nd.population = inds
        nd.fitness = res.F
        return nd, None, None
 
    return run

def make_nsga2(params, seed):
    return NSGA2(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

def make_spea2(params, seed):
    return SPEA2(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

def make_smsemoa(params, seed):
    return SMSEMOA(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

def make_moead(params, seed):
    return MOEAD(
        get_reference_directions("uniform", 2, n_partitions=params["population_size"]),
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

def make_nsde(params, seed):
    return NSDE(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

def make_kgb(params, seed):
    return KGB(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

def make_dnsga2(params, seed):
    return DNSGA2(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )
def make_gde3(params, seed):
    return GDE3(
        pop_size=int(params['population_size']),
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=params['crossover_prob']),
        mutation=BitflipMutation(prob=params['mutation_prob']),
        eliminate_duplicates=True
    )

# def make_agemoea2(params, seed):
#     return AGEMOEA2(
#         pop_size=params['population_size'],
#         eliminate_duplicates=True
#     )


