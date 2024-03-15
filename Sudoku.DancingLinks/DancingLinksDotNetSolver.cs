using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sudoku.Shared;
using DlxLib;

namespace Sudoku.DancingLinks
{
    public class DancingLinksDotNetSolver : ISudokuSolver
    {
        /// <summary>
        /// Solves the given Sudoku grid using a dancing links algorithm.
        /// </summary>
        /// <param name="s">The Sudoku grid to be solved.</param>
        /// <returns>
        /// The solved Sudoku grid.
        /// </returns>
        public SudokuGrid Solve(SudokuGrid inputSudoku)
        {

            var dlx = new Dlx();

            var matrix = Generate(inputSudoku);

            IEnumerable<Solution> solutions = dlx.Solve(matrix);

            return ConvertToSudokuGrid(solutions.First(), matrix);
        }


        private int[,] Generate(SudokuGrid inputSudoku)
        {
            int[,] dlxMatrix = new int[729, 324];
            int cellValue;
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    cellValue = inputSudoku.Cells[i, j];
                    if (cellValue != 0)
                    {
                        int[] knownRow = new int[324];
                        knownRow[i + j * 9] = 1;
                        knownRow[81 + j * 9 + cellValue - 1] = 1;
                        knownRow[81 * 2 + i * 9 + cellValue - 1] = 1;
                        knownRow[81 * 3 + ((i / 3) + (j / 3) * 3) * 9 + cellValue - 1] = 1;
                        for (int k = 0; k < 324; k++)
                        {
                            dlxMatrix[(i + j * 9) * 9, k] = knownRow[k];
                        }
                    }
                    else
                    {
                        for (int val = 1; val <= 9; val++)
                        {
                            cellValue = val;
                            int[] knownRow = new int[324];
                            knownRow[i + j * 9] = 1;
                            knownRow[81 + j * 9 + cellValue - 1] = 1;
                            knownRow[81 * 2 + i * 9 + cellValue - 1] = 1;
                            knownRow[81 * 3 + ((i / 3) + (j / 3) * 3) * 9 + cellValue - 1] = 1;
                            for (int k = 0; k < 324; k++)
                            {
                                dlxMatrix[(i + j * 9) * 9 + val - 1, k] = knownRow[k];
                            }
                        }
                    }
                }
            }
            return dlxMatrix;
        }

        private SudokuGrid ConvertToSudokuGrid(Solution solution, int[,] inputMatrix)
        {
            var sudokuGrid = new SudokuGrid(); //new int[9][];

            foreach (var row in solution.RowIndexes)
            {
                int cellId = -1;
                for (int i = 0; i < 81; i++)
                {
                    if (inputMatrix[row, i] == 1)
                    {
                        cellId = i;
                        break;
                    }
                }
                int cellValue = -1;
                for (int i = 81; i < 162; i++)
                {
                    if (inputMatrix[row, i] == 1)
                    {
                        cellValue = i % 9;
                        break;
                    }
                }
                sudokuGrid.Cells[cellId % 9, cellId / 9] = cellValue + 1;
            }

            return sudokuGrid;
        }
        
    }
}

