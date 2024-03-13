from timeit import default_timer
import numpy as np
import random
import pygad


random.seed(1)

instance = ((0,0,0,0,9,4,0,3,0),        
          (0,0,0,5,1,0,0,0,7),
          (0,8,9,0,0,0,0,4,0),
          (0,0,0,0,0,0,2,0,8),
          (0,6,0,2,0,1,0,5,0),
          (1,0,2,0,0,0,0,0,0),
          (0,7,0,0,0,0,5,2,0),
          (9,0,0,0,6,5,0,0,0),
          (0,4,0,9,7,0,0,0,0))

fixed_number = np.array(instance).reshape(9,9) != 0 

best_solution_fitness = 0

def generate_initial_sudoku(instance):
        instance = np.array(instance).flatten()      
        for i,e in enumerate(instance):
                if e == 0:
                        instance[i] = random.randint(1, 9)
       # print(type(instance))
       # print(np.array(instance).reshape(9,9))
        #print('toto')

        return instance

def create_initial_pop(instance,pop_size):
        population = [np.array(generate_initial_sudoku(instance)).flatten() for _ in range(pop_size)]
        return np.array(population)

def fitness_func(ga_instance, solution,solution_idx):
       # if solution_idx == 0:
       #         print(np.array(solution).reshape(9,9))
        fitness = 1000
        solution = np.array(solution).reshape(9,9)
        for row in solution: # row
                fitness-= 9 - len(np.unique(row))
        for col in solution.T: # col
                fitness-= 9 - len(np.unique(col))
        
        for i in range(0,9,3): #block
                for j in range(0,9,3):
                        block = []
                        for k in range(3):
                                for l in range(3):
                                        num = solution[i+k, j+l]
                                        if num != 0:
                                                block.append(num)
                        fitness-= 9 - len(set(block))
        global best_solution_fitness
        if fitness >= best_solution_fitness:
               fitness -= 1
               best_solution_fitness = fitness
        return fitness

def crossover_func(parents, offspring_size, ga_instance):
        offsprings = []
        crossover_probability = random.random()
        blocks = [parent.reshape(3,3,3,3).swapaxes(1,2).reshape(9,3,3) for parent in parents]
        for i in range(offspring_size[0]):
                offspring = blocks
                for j in range(9):
                        if crossover_probability < 0.5:
                                (offspring[0][j], offspring[1][j]) = (offspring[1][j], offspring[0][j])
                        crossover_probability = random.random()
                offsprings.append(offspring[0].reshape(3,3,3,3).swapaxes(1,2).reshape(9,9).flatten())
        return np.array(offsprings)
        

def mutation_func(offspring, ga_instance):
    mutation_probability = 0.1
    num_mutations = int(mutation_probability * offspring.shape[1]) 
    for idx in range(offspring.shape[0]):
        mutation_indices = np.random.choice(range(offspring.shape[1]), num_mutations, replace=False)
        for gene_index in mutation_indices:
            if not fixed_number.flatten()[gene_index]:
                new_val = np.random.choice(ga_instance.gene_space)
                offspring[idx, gene_index] = new_val
    return offspring

def on_generation(ga_instance):
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Generation #", ga_instance.generations_completed)
        print("Best solution so far:\n", np.array(solution).reshape(9,9))
        print("Best solution fitness:", solution_fitness)

# Params of the instance:

num_generations = 5000 # Nombre de generations
sol_per_pop = 20  # Nombre de solution par generations 
num_genes = 9 * 9 # Nombre de genes ici 81 car 81 cases dans un Sudoku
gene_space = [i for i in range(1, 10)]  # Valeurs que peuvent prendre les genes, ici de 1 -> 9 pour les valeurs possible dans un Sudoku
mutation_percent_genes = 10 # Pourcentage de mutations

# Autre parametres a rajouter #TODO

# Genetic Instance:

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       gene_space=gene_space,
                       initial_population = create_initial_pop(instance,sol_per_pop),
                       mutation_percent_genes=mutation_percent_genes,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func,
                       on_generation = on_generation
                       )

# Run Instance:
start = default_timer()

ga_instance.run()


# Get the best solution after the last generation
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Paramètres de la solution :\n", np.array(solution).reshape(9,9))
print("Fitness de la solution :", solution_fitness)

# if(solveSudoku(instance)):
# 	print_grid(instance)
# 	r=instance
# else:
# 	print ("Aucune solution trouv�e")

execution = default_timer() - start
print("Le temps de r�solution est de : ", execution, " seconds as a floating point value")