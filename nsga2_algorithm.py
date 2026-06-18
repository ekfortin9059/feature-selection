import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
 
import NSGA_II_functions as nsga2
from population_class import Population
from data_class import Data
from ML_code import eval_model
import benchmark_metrics as bm
import pandas as pd


#%% Load data
data = Data(464)

#%%# Parameters
N = 100
T_max = 20 
p_c = 0.8
p_m = 0.02

#%%
def my_NSGA_II(N, data, n_generations, p_c, p_m, seed=None):
    '''
    NSGA-II Main Loop
    '''
    
    rng = np.random.default_rng(seed)
    
    parent_pop = Population(N)
    parent_pop.initialize(data.n, rng)
    parent_pop.evaluate(data)
    
    fronts = nsga2.fast_non_dominated_sort(parent_pop)
    
    for front in fronts:
        nsga2.crowding_distance(front)
        
    for gen in range(n_generations):
        offspring_pop = Population(N)
        
        while len(offspring_pop.population) < N:
            
            # selection
            parent1 = nsga2.tournament_selection(parent_pop, rng)
            parent2 = nsga2.tournament_selection(parent_pop, rng)
            
            # crossover
            child1, child2 = nsga2.uniform_crossover(
                parent1, parent2, p_c, rng
                )            
            
            # mutation
            nsga2.mutation(child1, p_m, rng)
            nsga2.mutation(child2, p_m, rng)
            
            child1.evaluate_fitness(data)
            child2.evaluate_fitness(data)
            
            offspring_pop.population.append(child1)
            if len(offspring_pop.population) < N:
                offspring_pop.population.append(child2)
                        
        # elitist selection
        parent_pop = nsga2.generation_algorithm(parent_pop, offspring_pop, N)
                        
    return parent_pop
            
#%% Single run for testing
import time
start = time.time()
pop = my_NSGA_II(N, data, T_max, p_c, p_m, seed=100)
print(f"Time to run: {time.time() - start}")

fronts = nsga2.fast_non_dominated_sort(pop)
pareto_front = fronts[0]

fitness = np.array([ind.fitness for ind in pareto_front])

ref_point = np.array([1.1, data.n + 1]) # reference point encompasses entire solution space

hv_myNSGA2= HV(ref_point=ref_point)(fitness)
print(f"Hypervolume: {hv_myNSGA2:.4f}")

 
fitness_plot = fitness.copy()
fitness_plot[:, 0] = np.abs(fitness_plot[:, 0])  # convert back to original nonneg ML score
plt.figure(figsize=(7, 5))
plt.scatter(fitness_plot[:, 1], fitness_plot[:, 0], facecolors='none', edgecolors='blue')
plt.title("Objective Space - My NSGA-II")
plt.show()

#%% pymoo setup
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling


class myProblem(ElementwiseProblem):
    def __init__(self, data):
        super().__init__(
            n_var=data.n,        
            n_obj=2,             
            xl=0,                
            xu=1,                
            vtype=bool
        )
        self.data = data
        
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = -eval_model(self.data, X.astype(bool))
        f2 = np.sum(X)
        out["F"] = np.column_stack([f1, f2])

def make_pymoo_algorithm(p_c, p_m, N):
    return NSGA2(
        pop_size=N,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(prob=p_c),
        mutation=BitflipMutation(prob=p_m),
        eliminate_duplicates=True    )


#%% Single pymoo NSGA-II run
problem = myProblem(data)
algorithm = make_pymoo_algorithm(p_c, p_m, N)

res = minimize(
    problem,
    algorithm,
    termination=('n_gen', T_max),
    seed=100,
    verbose=False
)

F_pymoo = res.F  
hv_pymoo = HV(ref_point=ref_point)(F_pymoo)
print(f"Hypervolume (pymoo): {hv_pymoo:.4f}")
 
plt.figure(figsize=(7, 5))
plt.scatter(F_pymoo[:, 1], -F_pymoo[:, 0], facecolors='none', edgecolors='red')
plt.title("Objective Space - Pymoo NSGA-II")
plt.show()


#%% Pareto front extremes 
extreme_1, extreme_2 = bm.compute_pareto_extremes(data, eval_model)

#%% Metrics on multiple seeds: hypervolume, spread, spacing, and runtime 
seeds = range(20)
results = bm.run_benchmark(
    seeds, N, data, T_max, p_c, p_m, ref_point, extreme_1, extreme_2,
    my_nsga2_func=my_NSGA_II,
    pymoo_problem=myProblem,
    pymoo_algorithm=make_pymoo_algorithm,
    pymoo_minimize=minimize
)

#%% 
table = {}
for r in results:
    seeds = f"seed {r['seed']}     "
    
    table[seeds] = {
                    "": "my alg  pymoo |",
                    "Hypervolume": f"{r['my_HV']:.2f} {r['pymoo_HV']:.2f} |",
                    "Spread": f"{r['my_spread']:.2f}   {r['pymoo_spread']:.2f} |",
                    "Spacing": f"{r['my_spacing']:.2f}   {r['pymoo_spacing']:.2f} |",
                    "Runtime (s)": f"{r['my_time']:.2f}  {r['pymoo_time']:.2f} |"}
    
    df = pd.DataFrame(table)


print(df)












