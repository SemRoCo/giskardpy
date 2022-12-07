import gurobipy
import numpy as np
from gurobipy import GRB
from scipy import sparse

from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging

gurobipy.setParam('LogToConsole', False)

error_info = {
    gurobipy.GRB.LOADED: "Model is loaded, but no solution information is available.",
    gurobipy.GRB.OPTIMAL: "Model was solved to optimality (subject to tolerances), and an optimal solution is available.",
    gurobipy.GRB.INFEASIBLE: "Model was proven to be infeasible.",
    gurobipy.GRB.INF_OR_UNBD: "Model was proven to be either infeasible or unbounded. "
                              "To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.",
    gurobipy.GRB.UNBOUNDED: "Model was proven to be unbounded. "
                            "Important note: an unbounded status indicates the presence of an unbounded ray that allows "
                            "the objective to improve without limit. "
                            "It says nothing about whether the model has a feasible solution. "
                            "If you require information on feasibility, "
                            "you should set the objective to zero and reoptimize.",
    gurobipy.GRB.CUTOFF: "Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. "
                         "No solution information is available.",
    gurobipy.GRB.ITERATION_LIMIT: "Optimization terminated because the total number of simplex iterations performed "
                                  "exceeded the value specified in the IterationLimit parameter, or because the total "
                                  "number of barrier iterations exceeded the value specified in the BarIterLimit parameter.",
    gurobipy.GRB.NODE_LIMIT: "Optimization terminated because the total number of branch-and-cut nodes explored exceeded "
                             "the value specified in the NodeLimit parameter.",
    gurobipy.GRB.TIME_LIMIT: "Optimization terminated because the time expended exceeded the value specified in the "
                             "TimeLimit parameter.",
    gurobipy.GRB.SOLUTION_LIMIT: "Optimization terminated because the number of solutions found reached the value "
                                 "specified in the SolutionLimit parameter.",
    gurobipy.GRB.INTERRUPTED: "Optimization was terminated by the user.",
    gurobipy.GRB.NUMERIC: "Optimization was terminated due to unrecoverable numerical difficulties.",
    gurobipy.GRB.SUBOPTIMAL: "Unable to satisfy optimality tolerances; a sub-optimal solution is available.",
    gurobipy.GRB.INPROGRESS: "An asynchronous optimization call was made, but the associated optimization run is not "
                             "yet complete.",
    gurobipy.GRB.USER_OBJ_LIMIT: "User specified an objective limit (a bound on either the best objective or the best "
                                 "bound), and that limit has been reached.",
}


class QPSolverGurobi(QPSolver):
    STATUS_VALUE_DICT = {getattr(gurobipy.GRB.status, name): name for name in dir(gurobipy.GRB.status) if
                         '__' not in name}

    def __init__(self, num_non_slack: int, retry_added_slack: float, retry_weight_factor: float,
                 retries_with_relaxed_constraints: int):
        super().__init__(num_non_slack, retry_added_slack, retry_weight_factor, retries_with_relaxed_constraints)
        self.started = False

    @profile
    def init(self, H, g, A, lb, ub, lbA, ubA):
        self.qpProblem = gurobipy.Model('qp')
        self.x = self.qpProblem.addMVar(lb.shape, lb=lb, ub=ub)
        self.qpProblem.setMObjective(Q=H, c=None, constant=0.0, xQ_L=self.x, xQ_R=self.x, sense=GRB.MINIMIZE)
        # H = sparse.csc_matrix(H)
        A = sparse.csc_matrix(A)
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.LESS_EQUAL, ubA)
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.GREATER_EQUAL, lbA)
        self.started = False

    # def update(self, H, g, A, lb, ub, lbA, ubA):
    #     self.x.lb = lb
    #     self.x.ub = ub
    #     # self.qpProblem.setMObjective()
    #     self.qpProblem.remove(self.qpProblem.getConstrs())
    #     self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.LESS_EQUAL, ubA)
    #     self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.GREATER_EQUAL, lbA)
    #     self.qpProblem.setMObjective(H, None, 0.0)

    def print_debug(self):
        gurobipy.setParam('LogToConsole', True)
        logging.logwarn(error_info[self.qpProblem.status])
        self.qpProblem.reset()
        self.qpProblem.optimize()
        self.qpProblem.printStats()
        self.qpProblem.printQuality()
        gurobipy.setParam('LogToConsole', False)

    def round(self, data, decimal_places):
        return np.round(data, decimal_places)

    @profile
    def solve(self, weights: np.ndarray, g: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray, lbA: np.ndarray,
              ubA: np.ndarray) -> np.ndarray:
        H = np.diag(weights)
        self.init(H, g, A, lb, ub, lbA, ubA)
        self.qpProblem.optimize()
        success = self.qpProblem.status
        if success in {gurobipy.GRB.OPTIMAL, gurobipy.GRB.SUBOPTIMAL}:
            if success == gurobipy.GRB.SUBOPTIMAL:
                logging.logwarn('warning, suboptimal solution!')
            return np.array(self.qpProblem.X)
        if success == gurobipy.GRB.INFEASIBLE:
            raise InfeasibleException(self.STATUS_VALUE_DICT[success], success)
        raise QPSolverException(self.STATUS_VALUE_DICT[success], success)

    @profile
    def solve_and_retry(self, weights, g, A, lb, ub, lbA, ubA):
        exception = None
        for i in range(2):
            try:
                return self.solve(weights, g, A, lb, ub, lbA, ubA)
            except QPSolverException as e:
                exception = e
                # if e.error_code == gurobipy.GRB.NUMERIC:
                #     logging.logwarn(f'Solver returned \'{e}\', '
                #                     f'retrying with data rounded to \'{self.on_fail_round_to}\' decimal places')
                #     weights = self.round(weights, self.on_fail_round_to)
                #     A = self.round(A, self.on_fail_round_to)
                #     lb = self.round(lb, self.on_fail_round_to)
                #     ub = self.round(ub, self.on_fail_round_to)
                #     lbA = self.round(lbA, self.on_fail_round_to)
                #     ubA = self.round(ubA, self.on_fail_round_to)
                #     continue
                # if isinstance(e, InfeasibleException):
                try:
                    weights, lb, ub = self.compute_relaxed_hard_constraints(weights, g, A, lb, ub, lbA, ubA)
                    logging.loginfo(f'{e}; retrying with relaxed hard constraints')
                except InfeasibleException as e2:
                    if isinstance(e2, HardConstraintsViolatedException):
                        raise e2
                    raise e
                continue
        raise exception
