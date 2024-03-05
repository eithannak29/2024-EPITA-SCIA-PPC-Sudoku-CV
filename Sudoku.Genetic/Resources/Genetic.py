from timeit import default_timer
from tqdm import tqdm
import pygad


instance = ((0,0,0,0,9,4,0,3,0),
          (0,0,0,5,1,0,0,0,7),
          (0,8,9,0,0,0,0,4,0),
          (0,0,0,0,0,0,2,0,8),
          (0,6,0,2,0,1,0,5,0),
          (1,0,2,0,0,0,0,0,0),
          (0,7,0,0,0,0,5,2,0),
          (9,0,0,0,6,5,0,0,0),
          (0,4,0,9,7,0,0,0,0))

def fitness_func(solution,solution_idx):
        #TODO
        return;

def crossover_func():
        #TODO
        return;
        

def mutation_func():
        #TODO
        return;

# Params of the instance:

num_generations = 200 # Nombre de generations
sol_per_pop = 20  # Nombre de solution par generations 
num_genes = 9 * 9 # Nombre de genes ici 81 car 81 cases dans un Sudoku
gene_space = [i for i in range(1, 10)]  # Valeurs que peuvent prendre les genes, ici de 1 -> 9 pour les valeurs possible dans un Sudoku
mutation_percent_genes = 10, # Pourcentage de mutations

# Autre parametres a rajouter #TODO

# Genetic Instance:

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       mutation_percent_genes=mutation_percent_genes,
                       #crossover_type=crossover_func,
                       #mutation_type=mutation_func
                       )

# Run Instance:

tqdm(ga_instance.run())

# Get the best solution after the last generation
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Paramètres de la solution :", solution)
print("Fitness de la solution :", solution_fitness)

start = default_timer()
# if(solveSudoku(instance)):
# 	print_grid(instance)
# 	r=instance
# else:
# 	print ("Aucune solution trouv�e")

execution = default_timer() - start
print("Le temps de r�solution est de : ", execution, " seconds as a floating point value")