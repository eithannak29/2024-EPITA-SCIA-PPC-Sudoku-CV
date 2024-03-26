import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class SudokuDataset(Dataset):
    """Sudoku dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with puzzles.
        """
        self.sudoku_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.sudoku_frame)

    def __getitem__(self, idx):
        
        
        x = one_hot_encode(self.sudoku_frame.loc[idx].puzzle)
        y = one_hot_encode(self.sudoku_frame.loc[idx].solution)
        
        
        sample = {'x': x, 'y': y}
        return sample
    
def one_hot_encode(s):
    zeros = torch.zeros((81, 9), dtype=torch.float)
    for a in range(81):
        zeros[a, int(s[a]) - 1] = 1 if int(s[a]) > 0 else 0
    return zeros