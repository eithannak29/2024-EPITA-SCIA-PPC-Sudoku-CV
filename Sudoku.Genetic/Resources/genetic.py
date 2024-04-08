from timeit import default_timer
import numpy as np

import random
import pygad
import warnings
#import matplotlib.pyplot as plt
from utils import create_initial_pop,swap_blocks,matrix_difference,create_initial_pop_with_mask, generate_mask_from_instance, get_index_most_appear

warnings.filterwarnings("ignore", category=UserWarning)

# Hyper-Parameters

num_generations = 100 # Nombre de generations
sol_per_pop = 1000  # Nombre de solution par generations
gene_space = [i for i in range(1, 10)]  # Valeurs que peuvent prendre les genes, ici de 1 -> 9 pour les valeurs possible dans un Sudoku
mutation_percent_genes = 10 # Pourcentage de mutations
crossover_percent = 100 # Pourcentage de crossover

solution_nb : int = 1

instance : tuple[tuple] = ((0,0,0,0,9,4,0,3,0),
          (0,0,0,5,1,0,0,0,7),
          (0,8,9,0,0,0,0,4,0),
          (0,0,0,0,0,0,2,0,8),
          (0,6,0,2,0,1,0,5,0),
          (1,0,2,0,0,0,0,0,0),
          (0,7,0,0,0,0,5,2,0),
          (9,0,0,0,6,5,0,0,0),
          (0,4,0,9,7,0,0,0,0))

error_tab = [[0,0,0] for _ in range(sol_per_pop)]  #[col, row, fitness] for each solution in the population

# The fitness function apply a penalty for each duplicate in a row or column
def fitness_func(ga_instance : pygad.GA, solution : np.ndarray , solution_idx : int) -> float:
        fitness = 100
        solution = np.array(solution).reshape(9,9)
        for i in range(9):
                row_error = 9 - len(np.unique(solution[i, :])) #row
                col_error = 9 - len(np.unique(solution[:, i])) #column
                fitness -= (row_error + col_error)
                if row_error > error_tab[solution_idx][1] :
                     error_tab[solution_idx][1] = i  #set the worst row index
                if col_error > error_tab[solution_idx][0] :
                     error_tab[solution_idx][0] = i #set the worst column index
        error_tab[solution_idx][2] = fitness
        return fitness

# The crossover function is merge of blocks of two sudoku puzzles by splitting at a random point
def crossover_func(parents : np.ndarray, offspring_size : tuple , ga_instance : pygad.GA) -> np.ndarray:
    offsprings = []
    blocks = [parent.reshape(3,3,3,3).swapaxes(1,2).reshape(9,3,3) for parent in parents]
    for _ in range(offspring_size[0]):
        offspring = blocks
        crossover_point = random.randint(0, 9)
        offspring = swap_blocks(offspring, crossover_point)
        offsprings.append(offspring[0].reshape(3,3,3,3).swapaxes(1,2).reshape(9,9).flatten())
    return np.array(offsprings)


# Global Variables
default_mask = generate_mask_from_instance(instance)
# TODO
# mutation avec mask
# swap en fonction du l'erreur de la solution -> ou ce situe l'erreur

def mutation_error(offspring : np.ndarray, ga_instance : pygad.GA, idx : int, fixed_number):
    global default_mask, error_tab
    
    s = offspring[idx].reshape(9,9)
    i_worst_col = get_index_most_appear(s[:, error_tab[idx][0]])  #get index of error in the worst column
    j_worst_row = get_index_most_appear(s[error_tab[idx][1], :])  #get index of error in the worst row
    swap_idx = -1
    try_ = 0
    while try_ < 5:
        try_ += 1
        i_col = np.random.choice(len(i_worst_col), 1, replace=False)[0]  #select randomly one of the worst column
        j_row = np.random.choice(len(j_worst_row), 1, replace=False)[0]  #select randomly one of the worst row
        if np.random.random() <= 0.5 and not fixed_number[i_worst_col[i_col], error_tab[idx][0]]:  #swap with the worst column or the worst row
            swap_idx = i_worst_col[i_col]*9 + error_tab[idx][0]
            break
        elif not fixed_number[error_tab[idx][1], j_worst_row[j_row]]:
            swap_idx = error_tab[idx][1]*9 + j_worst_row[j_row]
            break
        else:
            continue
    if (swap_idx == -1):
        swap_idx = np.random.randint(0, 81)
    return swap_idx  #return the index of the error to swap

# the mutation function is a simple swap of two random cells in the same 3x3 grid of the Sudoku puzzle
def mutation_func(offspring : np.ndarray, ga_instance : pygad.GA) -> np.ndarray:
    global default_mask, error_tab

    fixed_number = np.array(instance).reshape(9,9) != 0
    mutation_probability = 1 
    num_mutations = int(mutation_probability * offspring.shape[0])

    for _ in range(num_mutations):
        idx = np.random.randint(offspring.shape[0])

        if (error_tab[idx][2] >= 95 ):  #if high fitness, mutation_error (mutation with error swap)
            swap_idx = mutation_error(offspring, ga_instance, idx, fixed_number)
            grid_x = (swap_idx // 9) // 3 * 3
            grid_y = (swap_idx % 9) // 3 * 3
            non_fixed_indices = []
            for i in range(grid_x, grid_x + 3):
                for j in range(grid_y, grid_y + 3):
                    if not fixed_number[i, j]:
                        non_fixed_indices.append((i, j))

            if len(non_fixed_indices) < 2:
                continue
            try_ = 0
            while try_ < 5:  #avoid infinite loop
                try_ += 1
                swap_indice = np.random.choice(len(non_fixed_indices), 1, replace=False) # Choose 1 random indices to swap
                i1, j1 = non_fixed_indices[swap_indice[0]]
                i2 = int(swap_idx // 9)
                j2 = swap_idx % 9 - 1
                if default_mask[int(offspring[idx, i1*9 + j1] - 1), i1, j1] and default_mask[int(offspring[idx, swap_idx] - 1), i2, j2] and (i1 != i2 or j1 != j2):       # Check if the swap is valid according to the mask -> only swap both values are True in the mask
                    offspring[idx, i1*9 + j1], offspring[idx, swap_idx] = offspring[idx, swap_idx], offspring[idx, i1*9 + j1]
                    break   # Exit the loop if the swap is done
        else:  #else random swap mutation
            grid_x = np.random.choice([0, 3, 6])
            grid_y = np.random.choice([0, 3, 6])

            non_fixed_indices = []
            for i in range(grid_x, grid_x + 3):
                for j in range(grid_y, grid_y + 3):
                    if not fixed_number[i, j]:
                        non_fixed_indices.append((i, j))

            if len(non_fixed_indices) < 2:
                continue

            try_ = 0
            while try_ < 5:  #avoid infinite loop
                try_ += 1
                swap_indices = np.random.choice(len(non_fixed_indices), 2, replace=False) # Choose 2 random indices to swap
                i1, j1 = non_fixed_indices[swap_indices[0]]
                i2, j2 = non_fixed_indices[swap_indices[1]]
                if default_mask[int(offspring[idx, i1*9 + j1] - 1), i1, j1] and default_mask[int(offspring[idx, i2*9 + j2] - 1), i2, j2]:       # Check if the swap is valid according to the mask -> only swap both values are True in the mask
                    offspring[idx, i1*9 + j1], offspring[idx, i2*9 + j2] = offspring[idx, i2*9 + j2], offspring[idx, i1*9 + j1]
                    break   # Exit the loop if the swap is done
                
                 
             
    return offspring

#allow_duplicate_genes try

previous_best_solution = None

def on_generation(ga_instance: pygad.GA) -> None:
    global previous_best_solution, fitness_values, lines

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Generation #", ga_instance.generations_completed)
    print("Best solution fitness:", solution_fitness)
    
    # global error_tab
    # print("print 10 first values of error_tab: ", error_tab[0:10])

#     fitness_values.append(100-solution_fitness)
#     lines.set_xdata(list(range(ga_instance.generations_completed)))
#     lines.set_ydata(fitness_values)
#     fig.canvas.draw()
#     fig.canvas.flush_events()

    if previous_best_solution is not None:
        diff = matrix_difference(np.array(previous_best_solution).reshape(9,9), np.array(solution).reshape(9,9))
        if np.all(diff == 0):
            print("The best solution has not changed from the previous generation.")
        else:
            print("The best solution has changed from the previous generation.")

    previous_best_solution = solution


# Initialize plotting
# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots()
# fitness_values = []
# lines, = ax.plot([], [], '-b')  # Returns a tuple of line objects, thus the comma
# ax.set_xlim(0, num_generations)
# ax.set_ylim(0, 100)  # Assuming fitness is in the range 0-100
# ax.set_xlabel('Generation')
# ax.set_ylabel('Best Fitness Score')
# ax.set_title('Evolution of Fitness Score')

# Create Genetic Algorithm Instance

ga_instance : pygad.GA = pygad.GA(num_generations=num_generations,
        num_parents_mating=2,
        fitness_func=fitness_func,
        gene_space=gene_space,
        initial_population = create_initial_pop_with_mask(instance,sol_per_pop),
        crossover_type=crossover_func,
        mutation_type=mutation_func,
        on_generation = on_generation,
        stop_criteria = ["saturate_30"],
        parallel_processing=["thread", 4]
        )

# Run Instance:

start = default_timer()

def solveSudoku(instance):
        solution_nb = 1
        while True:
                ga_instance.run()
                solution, solution_fitness, solution_idx = ga_instance.best_solution()
                print(f"Paramètres de la solution # {solution_nb}:\n", np.array(solution).reshape(9,9))
                print(f"Fitness de la solution # {solution_nb}:", solution_fitness)
                solution_nb += 1
                if solution_fitness == 100:
                        break
        solution = np.array(solution).reshape(9,9)
        solution = tuple(map(tuple, solution))
        return solution

if(solveSudoku(instance)):
	print_grid(instance)
	r=instance
        
else:
	print ("Aucune solution trouvée")

execution = default_timer() - start
print("Le temps de résolution est de : ", execution, " seconds as a floating point value")
