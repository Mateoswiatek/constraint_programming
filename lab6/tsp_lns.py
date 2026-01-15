# basic Adaptive Large Neighborhood Search for MiniZinc
# made by Mateusz Ślayński for the teaching purposes
from __future__ import annotations
from dataclasses import InitVar, dataclass
import math
import random
from naive_adaptive_lns import CPSolution, CPPartialSolution, CPSolver, NaiveAdaptiveLNS
from minizinc import Instance


@dataclass
class TSPPartialSolution(CPPartialSolution):
    solution: TSPSolution
    initial_route: list[int]

    def to_output(self) -> str:
        return f"initial_route = {self.initial_route};"

    def fix_instance(self, instance: Instance):
        instance["initial_route"] = self.initial_route


@dataclass
class TSPSolution(CPSolution[TSPPartialSolution]):
    route: list[int]
    total_distance: int

    # attribute required to be used by the MiniZinc library (stores objective in optimization problems)
    objective: int = 0
    # attribute required to be used by the MiniZinc library
    _output_item: InitVar[str | None] = None

    def to_output(self) -> str:
        return "\n".join(
            [f"route = {self.route};", f"total_distance = {self.total_distance};"]
        )

    @property
    def objective_value(self) -> int:
        return self.total_distance

    @property
    def should_minimize(self) -> bool:
        return True

    def relax(self, ratio: float) -> TSPPartialSolution:
        """
        This function should relax random {rate}% of the solution.
        We start with a full solution, copy it and put some zeroes in the copy.
        """
        fixed_route = self.route.copy()
        # Calculate how many variables to relax (at least 1)
        n = len(fixed_route)
        # We can relax indices 1 to n-1 (0-indexed), but not index 0 (route beginning stays fixed)
        num_to_relax = max(1, math.ceil(ratio * (n - 1)))
        # Choose random indices to relax from indices 1..n-1 (not the first one)
        indices_to_relax = random.sample(range(1, n), num_to_relax)
        # Set the chosen indices to 0 (relax them)
        for idx in indices_to_relax:
            fixed_route[idx] = 0
        return TSPPartialSolution(self, fixed_route)


class TSPSolver(CPSolver[TSPPartialSolution]):
    def __init__(
        self,
        problem_path: str,
        init_model_path: str,
        improve_model_path: str,
        solver_name: str = "gecode",
        processes: int = 1,
    ) -> None:
        super().__init__(
            problem_path, init_model_path, improve_model_path, solver_name, processes
        )
        self._initial_model.output_type = TSPSolution
        self._improve_model.output_type = TSPSolution
        self._initial_instance = Instance(self._solver, self._initial_model)
        self._improve_instance = Instance(self._solver, self._improve_model)

    def initial_solution(self) -> TSPSolution:
        """
        Finds the initial solution
        Returns a new TSP Solution
        """
        return self._initial_instance.solve(processes=self._processes).solution

    def improve_partial_assignment(
        self, partial_solution: TSPPartialSolution
    ) -> TSPSolution:
        """
        This function improves the given solution.
        """
        # the branch method creates a new "copy of the model instance"
        with self._improve_instance.branch() as opt:
            # then we set the initial_route
            partial_solution.fix_instance(opt)
            return opt.solve(processes=self._processes).solution


if __name__ == "__main__":
    # change path to solve different instance
    problem_path = "data/eil51.json"
    init_model = "tsp_init.mzn"
    improve_model = "tsp_improve.mzn"
    solver_name = "gecode"
    processes = 1
    solver = TSPSolver(problem_path, init_model, improve_model, solver_name, processes)
    lns = NaiveAdaptiveLNS(solver, adaption_timelimit_in_sec=20)
    lns.solve()
