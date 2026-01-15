"""Simple Solver for a Traveling Salesperson Problem (TSP) using or-tools routing solver"""

from dataclasses import dataclass
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import argparse
from io import IOBase


@dataclass
class TSPProblem:
    Nodes: int
    Dist: list[list[int]]


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("Traveling Salesman Problem via OrTools!")
    parser.add_argument("datafile", type=argparse.FileType("r"), help="input json file")
    return parser.parse_args()


def load_problem(file: IOBase):
    """Loads problem parameters from a JSON file"""
    problem_dictionary = json.load(file)
    file.close()
    return TSPProblem(**problem_dictionary)


def create_model_structure(
    problem: TSPProblem,
) -> tuple[pywrapcp.RoutingModel, pywrapcp.RoutingIndexManager]:
    """Defines structure of the model"""
    index_manager = pywrapcp.RoutingIndexManager(  # RoutingIndexManager is responsible for translating nodes
        # to the variables in the model
        problem.Nodes,  # How many nodes there are
        1,  # How many vehicles there are
        0,  # Start point / Depot
    )

    # Create Routing Model.
    model = pywrapcp.RoutingModel(index_manager)
    return model, index_manager


def add_objective(
    problem: TSPProblem,
    model: pywrapcp.RoutingModel,
    index_manager: pywrapcp.RoutingIndexManager,
):
    """Add objective to the model"""

    # registers a function calculating distance between nodes
    transit_callback_index = model.RegisterTransitMatrix(problem.Dist)

    # defines the objective to be distance
    model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


def solve(
    problem: TSPProblem,
    model: pywrapcp.RoutingModel,
    index_manager: pywrapcp.RoutingIndexManager,
    args,
):
    """Solve the given model with default parameters"""

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )  # defines a solving strategy
    search_parameters.time_limit.FromSeconds(
        5
    )  # we need a timelimit or the guided local search may never stop

    # Solve the problem.
    return model.SolveWithParameters(search_parameters)


def print_solution(solution, model, index_manager):
    """Prints solution on console."""

    # Print solution on console.
    if solution is None:
        print("no solution at all")
        return

    print(f"total distance: {solution.ObjectiveValue()}")
    index = model.Start(0)  # we get the variable corresponding to the route start
    plan_output = ""
    while not model.IsEnd(index):
        plan_output += f" {index_manager.IndexToNode(index)} ->"  # IndexToNode translates internal variable to our node index
        index = solution.Value(
            model.NextVar(index)
        )  # Value gets a value, NextVar get the next variable in the route
    plan_output += (
        f" {index_manager.IndexToNode(index)}\n"  # Let's not forget about the last node
    )
    print(plan_output)


if __name__ == "__main__":
    args = parse_args()  # 1. parse args
    problem = load_problem(args.datafile)  # 2. load problem
    model, index_manager = create_model_structure(problem)  # 3. create routing model
    add_objective(problem, model, index_manager)  # 4. add distance objective
    solution = solve(problem, model, index_manager, args)  # 5. solve!
    print_solution(solution, model, index_manager)  # 6. print solution!
