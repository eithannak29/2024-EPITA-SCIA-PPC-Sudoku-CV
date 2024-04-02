using Sudoku.Shared;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Python.Deployment;
using Python.Runtime;
namespace Sudoku.NeuralNetwork;

public class NeuralNetSolver : PythonSolverBase
{
    public override SudokuGrid Solve(SudokuGrid s)
    {
        using (Py.GIL())
        {
            var pyScript = Py.Import("../Sudoku.NeuralNetwork/9millions/main.py");
            var message = new PyString(s.ToString());
            var result = pyScript.InvokeMethod("test", message);
            Console.WriteLine(result);
        }
        return s;
    }
    
   
}