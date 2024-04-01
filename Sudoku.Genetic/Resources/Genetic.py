from timeit import default_timer
import numpy as np
import random
import pygad

#random.seed(1)

solution_nb = 1

# instance = ((0,0,0,0,9,4,0,3,0),        
#           (0,0,0,5,1,0,0,0,7),
#           (0,8,9,0,0,0,0,4,0),
#           (0,0,0,0,0,0,2,0,8),
#           (0,6,0,2,0,1,0,5,0),
#           (1,0,2,0,0,0,0,0,0),
#           (0,7,0,0,0,0,5,2,0),
#           (9,0,0,0,6,5,0,0,0),
#           (0,4,0,9,7,0,0,0,0))

instance = ((0,8,0,0,0,0,0,9,0),
        (0,0,7,5,0,2,8,0,0),
        (6,0,0,8,0,7,0,0,5),
        (3,7,0,0,8,0,0,5,1),
        (2,0,0,0,0,0,0,0,8),
        (9,5,0,0,4,0,0,3,2),
        (8,0,0,1,0,4,0,0,9),
        (0,0,1,9,0,3,6,0,0),
        (0,4,0,0,0,0,0,2,0))

best_solution_fitness = 0


def generate_initial_sudoku(instance):
    instance = np.array(instance).flatten()
    for i in range(0, len(instance), 9):
        for j in range(i, i + 9):
            if instance[j] == 0:
                grid_x, grid_y = (j // 27) * 3, ((j % 9) // 3) * 3
                grid = instance[grid_x*9+grid_y : grid_x*9+grid_y+3] 
                grid = np.append(grid, instance[(grid_x+1)*9+grid_y : (grid_x+1)*9+grid_y+3])
                grid = np.append(grid, instance[(grid_x+2)*9+grid_y : (grid_x+2)*9+grid_y+3])
                possible_numbers = [n for n in range(1, 10) if n not in grid]
                
                if not possible_numbers:
                    instance[i:i+9] = [0 if instance[k] == 0 else instance[k] for k in range(i, i+9)]
                    break
                
                instance[j] = random.choice(possible_numbers)
    return instance

def create_initial_pop(instance: tuple[tuple],pop_size):
        population = [np.array(generate_initial_sudoku(instance)).flatten() for _ in range(pop_size)]
        return np.array(population)

def fitness_func(ga_instance, solution,solution_idx):
       # if solution_idx == 0:
       #         print(np.array(solution).reshape(9,9))
        fitness = 100
        solution = np.array(solution).reshape(9,9)
        for i in range(9):
                fitness-= 9 - len(np.unique(solution[i, :])) #row
                fitness-= 9 - len(np.unique(solution[:, i])) #column
        return fitness

def crossover_func(parents, offspring_size, ga_instance):
        offsprings = []
        blocks = [parent.reshape(3,3,3,3).swapaxes(1,2).reshape(9,3,3) for parent in parents]
        for i in range(offspring_size[0]):
                offspring = blocks
                crossover_point = random.randint(0, 9)
                for j in range(9):
                        if j > crossover_point:
                                (offspring[0][j], offspring[1][j]) = (offspring[1][j], offspring[0][j])
                offsprings.append(offspring[0].reshape(3,3,3,3).swapaxes(1,2).reshape(9,9).flatten())
        return np.array(offsprings)
        

# def mutation_func(offspring, ga_instance):
#     mutation_probability = 0.1
#     fixed_number = np.array(instance).reshape(9,9) != 0 
#     num_mutations = int(mutation_probability * offspring.shape[1]) 
#     for idx in range(offspring.shape[0]):
#         mutation_indices = np.random.choice(range(offspring.shape[1]), num_mutations, replace=False)
#         for gene_index in mutation_indices:
#             if not fixed_number.flatten()[gene_index]:
#                 new_val = np.random.choice(ga_instance.gene_space)
#                 offspring[idx, gene_index] = new_val
#     return offspring

def mutation_func(offspring, ga_instance):
    mutation_probability = 0.5
    fixed_number = np.array(instance).reshape(9,9) != 0 
    num_mutations = int(mutation_probability * offspring.shape[0]) 

    for _ in range(num_mutations):
        # Select a random offspring
        idx = np.random.randint(offspring.shape[0])

        # Select a random 3x3 grid
        grid_x = np.random.choice([0, 3, 6])
        grid_y = np.random.choice([0, 3, 6])

        # Get the indices of the non-fixed numbers in the grid
        non_fixed_indices = []
        for i in range(grid_x, grid_x + 3):
            for j in range(grid_y, grid_y + 3):
                if not fixed_number[i, j]:
                    non_fixed_indices.append((i, j))

        # If there are less than 2 non-fixed numbers, we can't swap anything
        if len(non_fixed_indices) < 2:
            continue

        # Select two random non-fixed numbers to swap
        swap_indices = np.random.choice(len(non_fixed_indices), 2, replace=False)
        i1, j1 = non_fixed_indices[swap_indices[0]]
        i2, j2 = non_fixed_indices[swap_indices[1]]

        # Swap the numbers
        offspring[idx, i1*9 + j1], offspring[idx, i2*9 + j2] = offspring[idx, i2*9 + j2], offspring[idx, i1*9 + j1]

    return offspring

def on_generation(ga_instance):
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Generation #", ga_instance.generations_completed, "Solution #", solution_nb)
        print("Best solution so far:\n", np.array(solution).reshape(9,9))
        print("Best solution fitness:", solution_fitness)

def init_gene_space(instance):
        gene_space = []
        all_values = [i for i in range(1,10)]
        initial_sudoku = np.array(instance).flatten()
        for x in initial_sudoku:
                if x == 0:
                       gene_space.append(all_values)
                else:
                       gene_space.append([x])
        return gene_space

# Params of the instance:

num_generations = 100 # Nombre de generations
sol_per_pop = 10000  # Nombre de solution par generations 
num_genes = 9 * 9 # Nombre de genes ici 81 car 81 cases dans un Sudoku
gene_space = [i for i in range(1, 10)]  # Valeurs que peuvent prendre les genes, ici de 1 -> 9 pour les valeurs possible dans un Sudoku
mutation_percent_genes = 10 # Pourcentage de mutations


# Genetic Instance:

# ga_instance = pygad.GA(num_generations=num_generations,
#                       num_parents_mating=2,
#                       fitness_func=fitness_func,
#                       gene_space=gene_space,
#                       initial_population = create_initial_pop(instance,sol_per_pop),
#                       #mutation_percent_genes=mutation_percent_genes,
#                       crossover_type='uniform',
#                       mutation_type=mutation_func,
#                       on_generation = on_generation,
#                       stop_criteria = ["saturate_10"],
#                       parallel_processing=["thread", 4]
#                       )

# Run Instance:
start = default_timer()

while True:
        ga_instance = pygad.GA(num_generations=num_generations,
                      num_parents_mating=2,
                      fitness_func=fitness_func,
                      gene_space=gene_space,
                      initial_population = create_initial_pop(instance,sol_per_pop),
                      #mutation_percent_genes=mutation_percent_genes,
                      crossover_type=crossover_func,
                      mutation_type=mutation_func,
                      on_generation = on_generation,
                      stop_criteria = ["saturate_30"],
                      parallel_processing=["thread", 4]
                      )
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Paramètres de la solution # {solution_nb}:\n", np.array(solution).reshape(9,9))
        print(f"Fitness de la solution # {solution_nb}:", solution_fitness)
        solution_nb += 1
        if solution_fitness == 400:
               break

# if(solveSudoku(instance)):
# 	print_grid(instance)
# 	r=instance
# else:
# 	print ("Aucune solution trouv�e")

execution = default_timer() - start
print("Le temps de r�solution est de : ", execution, " seconds as a floating point value")