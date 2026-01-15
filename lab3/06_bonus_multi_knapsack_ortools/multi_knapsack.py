import argparse
from dataclasses import dataclass
from fileinput import FileInput
from ortools.sat.python import cp_model
import json
import sys
import numpy as np
from numpy.typing import NDArray, ArrayLike

"""
This file is using ortools solver to solve the multi-knapsack problem.
Basically, it does the same thing as the multi_knapsack.mzn, but in python,
limited only to the ortools solver.

The first and most important issue is that ortools does not have the lexless_eq
global constraint! One has to implement by themselves...
"""

# to simulate integer domains
MAX_INT32 = 2**32

@dataclass
class Problem:
    """
    This class is responsible for holding problem parameters
    """
    items_n: int
    knapsacks_n: int
    capacity: int
    
    values: list[int]
    weights: list[int]

@dataclass
class Variables:
    """
    This class hold all decision variables in the model
    """
    taken: NDArray[cp_model.IntVar]
    total_value: cp_model.IntVar

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """ This callback print a solution when it is found """

    def __init__(self, variables: Variables, problem: Problem):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables = variables
        self.problem = problem
        self.solution_count = 0

    def on_solution_callback(self):
        """ It simulates the MiniZinc output """
        self.solution_count += 1
        for k, items in enumerate(self.variables.taken):
            print(f"{k+1}) {''.join([f'{self.Value(i)}' for i in items])}")
        print("----------")

    def solution_count(self):
        return self.solution_count

def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser("Constraint Programming via OrTools!")
    parser.add_argument('datafile', 
                        type = argparse.FileType('r'), 
                        help = "input json file")
    parser.add_argument('-a', '--all_solutions', 
                        action = "store_true",
                        dest = "enumerate_all_solutions",
                        help = "whether to print all the solutions")
    return parser.parse_args()

def load_problem(file: FileInput):
    """ Loads problem parameters from a JSON file """
    problem_dictionary = json.load(file)
    file.close()
    return Problem(**problem_dictionary)

def create_variables(problem: Problem, model: cp_model.CpModel):
    """ For each menu item, creates an integer variable with domain (-2^32, 2^32) """
    taken = np.array([
        [ model.NewBoolVar(f"taken_{k}_{i}") for k in range(problem.items_n) ] 
        for i in range(problem.knapsacks_n)])
    total_value = model.NewIntVar(0, sum(problem.values), "total_value")
    return Variables(taken, total_value)

def add_constraints(variables: Variables, problem: Problem, model: cp_model.CpModel):
    """ Adds constraints to the model """

    # item can be in one knapsack at most
    for i in range(problem.items_n):
        model.Add(sum(variables.taken[:,i]) <= 1)
    
    # capacity has to be respected
    for k in range(problem.knapsacks_n):
        model.Add(sum([variables.taken[k,i] * problem.weights[i] for i in range(problem.items_n)]) <= problem.capacity)
    
    model.Add(variables.total_value == sum([variables.taken[k,i] * problem.values[i] for k in range(problem.knapsacks_n) for i in range(problem.items_n)]))

    for k in range(problem.knapsacks_n - 1):
        add_lex_less_eq(model, variables.taken[k+1], variables.taken[k], "taken")


def add_lex_less_eq(model, left: ArrayLike, right: ArrayLike, name: str):
    '''
        Enforce `left` be to lexicographically less or equal compared to `right`
        `name` is the name of the constraint — it's not important, it is only useful for debugging purpose.
        You may use it to name the variables/constraints craeted here... 

        There is no such a global constraint built-in in or-tools, so we have to create it
        using normal constraints.  
    '''
    
    assert len(left) == len(right), "code works only for the lists of the same size"
    length = len(left)
    assert length > 0, "empty lists are not supported"

    if length == 1: # pathologic case with only one element in the list
        model.Add(left[0] <= right[0])
        return
    
    # OK. Let's do it!
    # 1) for every index `i` of the lists: `left` and `right`
    #    we need 3 boolean vars :
    #   - lt: `left[i] < right[i]`
    #   - eq: `left[i] == right[i]`
    #   - gt: `left[i] > right[i]` 
    # We will store them in the lists `lt_vars`, etc.  
    lt_vars = []
    eq_vars = []
    gt_vars = []

    # Then we have to iterate over all the elements of the lists
    for i, (l,r) in enumerate(zip(left, right)):
        # Now we have to do some manual reification!
        # We will say that the constraint `c` has to be satisfied only if the `v` variable is true.
        # This will be the "positive case":
        #      `model.Add(c).OnlyEnforceIf(v)`
        # To make the reification complete, we should also add the negative case.
        # Assuming that constraint `c'` is satisfied only if c is violated, we should write: 
        #      `model.Add(c').OnlyEnforceIf(v.Not())`
        # This would make sure that variable `v` is true only when `c` is satisfied and false otherwise.
        # For example:
        #       `model.Add(x1 < 5).OnlyEnforceIf(b1)`
        #       `model.Add(x1 >= 5).OnlyEnforceIf(b1.Not())`
        # Would reify the `x1 < 5` constraint into variable `b1`.
        # Luckily sometimes there is a simpler way as you will see in a second
        #   
        # TODO:
        # 1) create three new boolean variables (`lt_var`, `eq_var`, `gt_var`,and put them in the corresponding lists
        # 2) make sure that the constraints `l < r`, `l == r`, `l > r` are enforced only when the coressponding variables are true
        #    It's just the positive part of the reification.
        # 3) as these constraints are exclusive, you do not need the negative side
        #    it is enough to make sure that only one of the variables is true (`lt_var + eq_var + gt_var == 1`)
        #    
        pass
        

    # TODO: create a new boolean variable, storing result whether `left` is `lex less eq` than `right`    
    result_var = None

    # Special case: 
    # If `left == right`, then the `result_var == 1`
    #
    # TODO:
    # 1) create an extra boolean variable `all_eq`
    # 2) add constraint that `all_eq == eq_vars[0] && eq_vars[1] && ...`
    #   tip. `AddMinEquality(x, [y1,y2,y2])`` can be used to force `x == y1 && y2 && y3`
    # 3) add implication: `all_eq -> result_var`
    #  tip. model.AddImplication
    pass

    # Other cases, `left < right` or `left > right`
    # Now we go through all the lt_vars, eq_vars, gt_vars
    for i in range(length):
        # We need to make sure the `result_var` is true only when correct conditions are satisfied:
        # - assuming all elements with indices `< i` are equal in `left` and `right`:
        #   - if element `left[i] < right[i]` then `left < right`, so `result_var == 1`
        #   - if element `left[i] > right[i]` then `left > right`, so `result_var == 0`
        # In other words we compare prefixes of the lists: `left[:i+1]` and `right[:i+1]`
        #   - if `left[:i+1] < right[:i+1]` then `left < right`, so `result_var == 1`
        #   - if `left[:i+1] > right[:i+1]` then `left > right`, so `result_var == 0` 
        # TODO:
        # 1) get all `eq_vars` variables up to `i` (use slicing `eq_vars[:i]`)
        # 2) get `lt` and `gt` vars for the index `i` 
        # 3) create two boolean variables:
        #   - one to represent the scenario that `left[:i+1] < right[:i+1]`,
        #     let's name it `prefix_lt`
        #   - second one to represent the scenario that `left[:i+1] > right[:i+1]`
        #     let's name it `prefix_gt`
        # 4) add constraints:
        #   `prefix_lt = eq_vars[0] && ... && eq_vars[i-1] && lt[i]`
        #   `prefix_le = eq_vars[0] && ... && eq_vars[i-1] && gt[i]`
        # 5) add implications (`AddImplication`)
        #   - `prefix_lt -> result_var`
        #   - `prefix_gt -> result.var.Not()`
        pass
    
    # TODO: add constraint that result_var should be equal 1, i.e., the `left` has to be `lex less eq` than `right`  
    pass


def add_objective(variables: Variables, problem: Problem, model: cp_model.CpModel):
    """ Add objective to the model (original problem does not have any) """
    model.Maximize(variables.total_value)

def solve(variables: Variables, problem: Problem, model: cp_model.CpModel, args):
    """ Solve the given model"""

    # create a solver
    solver = cp_model.CpSolver()
    # create a solution printer
    solution_printer = SolutionPrinter(variables, problem)
    
    # set, based on the args, whether to enumerate all solutions
    solver.parameters.enumerate_all_solutions = args.enumerate_all_solutions

    # solve!
    status = solver.Solve(model, solution_printer)
    
    # print final status as in MiniZinc
    match status:
        case cp_model.OPTIMAL:
            print("==========")
        case cp_model.FEASIBLE:
            print("----------")
        case cp_model.INFEASIBLE:
            print("INFEASIBLE")
        case cp_model.MODEL_INVALID:
            print("INVALID")
        case cp_model.UNKNOWN:
            print("UNKOWN")


if __name__ == "__main__":
    args = parse_args()                          # 1. parse args
    problem = load_problem(args.datafile)        # 2. load problem
    model = cp_model.CpModel()                   # 3. create CP model
    variables = create_variables(problem, model) # 4. create variables
    add_constraints(variables, problem, model)   # 5. add constraints
    add_objective(variables, problem, model)     # 6. add objective
    solve(variables, problem, model, args)       # 7. solve!
