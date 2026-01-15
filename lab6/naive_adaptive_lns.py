from __future__ import annotations
from abc import ABC, abstractmethod
import time
from typing_extensions import Self
from minizinc import Instance, Model, Solver


class CPPartialSolution(ABC):
    """A partial solution to the problem
    Some variables are fixes, but other are left to be optimized
    """

    @abstractmethod
    def to_output(self) -> str:
        """just print the solution"""

    @abstractmethod
    def fix_instance(self, instance: Instance):
        """fixes fields in the solver input"""


class CPSolution[P : CPPartialSolution](ABC):
    """Full solution to the given problem."""

    @property
    @abstractmethod
    def objective_value(self) -> int:
        """objective value"""

    @abstractmethod
    def to_output(self) -> str:
        """just serializes the solution to string"""

    @property
    @abstractmethod
    def should_minimize(self) -> bool:
        """returns True, if we minimize objective, otherwise False"""

    @abstractmethod
    def relax(self, ratio: float) -> P:
        """return partial solution with {ratio} of variables relaxed"""

    def is_better_than(self, other: Self) -> bool:
        if self.should_minimize:
            return self.objective_value < other.objective_value
        return self.objective_value > other.objective_value


class CPSolver[P: CPPartialSolution](ABC):
    _initial_model: Model
    _improve_model: Model
    _solver: Solver
    _processes: int

    def __init__(
        self,
        problem_path: str,
        init_model_path: str,
        improve_model_path: str,
        solver_name: str = "gecode",
        processes: int = 1,
    ) -> None:
        super().__init__()

        # load minizinc solver
        self._solver = Solver.lookup(solver_name)
        self._processes = processes
        # load model finding the initial solution
        self._initial_model = Model()
        self._initial_model.add_file(init_model_path)
        self._initial_model.add_file(problem_path)

        # load model improving the solution
        self._improve_model = Model()
        self._improve_model.add_file(improve_model_path)
        self._improve_model.add_file(problem_path)

    @abstractmethod
    def initial_solution(self) -> CPSolution[P]:
        """finds initial feasible solution and its objective value"""

    @abstractmethod
    def improve_partial_assignment(self, partial_solution: P) -> CPSolution[P]:
        """improves the given solution using specified relax ratio"""


class NaiveAdaptiveLNS[P: CPPartialSolution]:
    solver: CPSolver[P]
    initial_zeroing_ratio: float
    adaption_rate: float
    adaption_timelimit_in_sec: int

    def __init__(
        self,
        solver: CPSolver[P],
        initial_zeroing_ratio: float = 0.1,
        adaption_rate: float = 0.05,
        adaption_timelimit_in_sec: int = 10,
    ) -> None:
        self.solver = solver
        self.initial_zeroing_ratio = initial_zeroing_ratio
        self.adaption_rate = adaption_rate
        self.adaption_timelimit_in_sec = adaption_timelimit_in_sec

    def solve(self):
        zeroing_ratio = self.initial_zeroing_ratio

        # just to calculate how much time we spent on the optimization
        checkpoint = time.time()
        # we get the initial solution and mark it as the best
        best_solution = self.solver.initial_solution()
        print("initial solution:")
        print(f"{best_solution.to_output()}")
        print("-------------------------")
        # the LNS loop, so exciting
        while zeroing_ratio <= 1.0:
            # we relax the current solution to get a relaxed assignment
            partial_solution = best_solution.relax(zeroing_ratio)
            # we try to optimize starting from partial assignment
            new_solution = self.solver.improve_partial_assignment(partial_solution)

            # if it's better than the old one
            if new_solution.is_better_than(best_solution):
                checkpoint = time.time()
                # we reset the zeroing rate
                zeroing_ratio = self.initial_zeroing_ratio
                # we remember the best solution
                best_solution = new_solution
                # and print it
                print("assignment:")
                print(f"{best_solution.to_output()}")
                print(f"objective = {best_solution.objective_value}")
                print("-------------------------")

            # if the solver struggles we increase the zeroing rate :)
            if time.time() - checkpoint > self.adaption_timelimit_in_sec:
                checkpoint = time.time()
                zeroing_ratio += self.adaption_rate
                print(f"* changed zeroing rate to {zeroing_ratio}")
