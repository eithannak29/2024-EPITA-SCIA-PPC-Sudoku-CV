import numpy as np
import random

def generate_initial_sudoku(instance: tuple[tuple]) -> np.ndarray:
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
    

def change_blocks(parents):
        blocks = [parent.reshape(3,3,3,3).swapaxes(1,2).reshape(9,3,3) for parent in parents]
        offspring = blocks
        crossover_point = random.randint(0, 9)
        for j in range(9):
                if j > crossover_point:
                        (offspring[0][j], offspring[1][j]) = (offspring[1][j], offspring[0][j])
        return np.array(offspring[0].reshape(3,3,3,3).swapaxes(1,2).reshape(9,9).flatten())
    
def swap_blocks(offspring, crossover_point):
    for j in range(9):
        if j > crossover_point:
            (offspring[0][j], offspring[1][j]) = (offspring[1][j], offspring[0][j])
    return offspring

