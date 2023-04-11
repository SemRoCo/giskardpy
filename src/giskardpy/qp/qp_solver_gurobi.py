from typing import Iterable, Tuple

import gurobipy
import numpy as np
from gurobipy import GRB
from scipy import sparse

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver_qpswift import QPSWIFTFormatter
from giskardpy.utils import logging
from giskardpy.utils.utils import record_time

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


class QPSolverGurobi(QPSWIFTFormatter):
    solver_id = SupportedQPSolver.gurobi
    STATUS_VALUE_DICT = {getattr(gurobipy.GRB.status, name): name for name in dir(gurobipy.GRB.status) if
                         '__' not in name}
    sparse = True
    compute_nI_I = False

    @profile
    def init(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, lb: np.ndarray,
             ub: np.ndarray, h: np.ndarray):
        self.qpProblem = gurobipy.Model('qp')
        self.x = self.qpProblem.addMVar(H.shape[0], lb=lb, ub=ub)
        H = sparse.diags(H, 0)
        self.qpProblem.setMObjective(Q=H, c=None, constant=0.0, xQ_L=self.x, xQ_R=self.x, sense=GRB.MINIMIZE)
        self.qpProblem.addMConstr(E, self.x, gurobipy.GRB.EQUAL, b)
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.LESS_EQUAL, h)
        self.started = False

    def print_debug(self):
        gurobipy.setParam('LogToConsole', True)
        logging.logwarn(error_info[self.qpProblem.status])
        self.qpProblem.reset()
        self.qpProblem.optimize()
        self.qpProblem.printStats()
        self.qpProblem.printQuality()
        gurobipy.setParam('LogToConsole', False)

    @profile
    def problem_data_to_qp_format(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lb, ub = self.lb_ub_with_inf(self.nlb, self.ub)
        return self.weights, self.g, self.E, self.bE, self.nA_A, lb, ub, self.nlbA_ubA

    @record_time
    @profile
    def solver_call(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, lb: np.ndarray,
                    ub: np.ndarray, h: np.ndarray) -> np.ndarray:
        self.init(H, g, E, b, A, lb, ub, h)
        self.qpProblem.optimize()
        success = self.qpProblem.status
        if success in {gurobipy.GRB.OPTIMAL, gurobipy.GRB.SUBOPTIMAL}:
            if success == gurobipy.GRB.SUBOPTIMAL:
                logging.logwarn('warning, suboptimal solution!')
            return np.array(self.qpProblem.X)
        if success in {gurobipy.GRB.INFEASIBLE, gurobipy.GRB.INF_OR_UNBD}:
            raise InfeasibleException(self.STATUS_VALUE_DICT[success], success)
        raise QPSolverException(self.STATUS_VALUE_DICT[success], success)
