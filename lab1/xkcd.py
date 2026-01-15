import argparse
from dataclasses import dataclass
from fileinput import FileInput
from ortools.sat.python import cp_model
import json
import sys

"""
This file is using ortools solver to solve the xkcd problem.
Basically, it does the same thing as the xkcd.mzn, but in python,
limited only to the ortools solver.

Still. It may be more suitable for deployment...
"""

# to simulate integer domains
MAX_INT32 = 2**32

@dataclass
class Problem:
    """
    This class is responsible for holding problem parameters
    """
    menu_length: int
    money_limit: int
    menu_prices: list[int]
    menu_names: list[str]

@dataclass
class Variables:
    """
    This class hold all decision variables in the model
    """
    order: list[cp_model.IntVar]

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
        for i, v in enumerate(self.variables.order):
            print(f"{self.problem.menu_names[i]}: {self.Value(v)}")
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
    return Problem(problem_dictionary["menu_length"],
                   problem_dictionary["money_limit"],
                   problem_dictionary["menu_prices"],
                   problem_dictionary["menu_names"])

def create_variables(problem: Problem, model: cp_model.CpModel):
    """ For each menu item, creates an integer variable with domain (-2^32, 2^32) """
    order = [ model.NewIntVar(-MAX_INT32, MAX_INT32, f"order_{i}") for i in range(problem.menu_length) ]
    return Variables(order)

def add_constraints(variables: Variables, problem: Problem, model: cp_model.CpModel):
    """ Adds constraints to the model """

    # every variable has to be positive
    for variable in variables.order:
        model.Add(variable >= 0)
    
    # creates auxiliary variables to hold price for each ordered item
    price_variables = [ model.NewIntVar(-MAX_INT32, MAX_INT32, f"price_{i}") for i in range(problem.menu_length) ]
    
    # sets value of a price value to a number of order items times their prices
    for order, price, price_var in zip(variables.order, problem.menu_prices, price_variables):
        model.Add(price_var == order * price)
    
    # creates an auxiliary variable holding total price od the order
    total_price = model.NewIntVar(-MAX_INT32, MAX_INT32, f"total_price")

    # the total price should equal sum of all the individual prices
    model.Add(total_price == sum(price_variables))

    # as the original problem states, the total prices has to equal to the limit
    model.Add(total_price == problem.money_limit)


def add_objective(variables: Variables, problem: Problem, model: cp_model.CpModel):
    """ Add objective to the model (original problem does not have any) """
    # model.maximize(sum(variables.orders)) would maximize number of ordered meals
    pass 

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
