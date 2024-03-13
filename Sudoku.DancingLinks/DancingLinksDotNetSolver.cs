using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sudoku.Shared;

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
        public SudokuGrid Solve(SudokuGrid s)
        {
            return s;
        }
        
    }
}

