from collections import defaultdict
from typing import Iterable, Tuple, Dict

import gurobipy
import numpy as np
from gurobipy import GRB
from gurobipy.gurobipy import GurobiError
from scipy import sparse

from giskardpy.configs.data_types import SupportedQPSolver
from giskardpy.exceptions import QPSolverException, InfeasibleException, HardConstraintsViolatedException
from giskardpy.qp.qp_solver import record_solver_call_time, QPSWIFTFormatter
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


class QPSolverGurobi(QPSWIFTFormatter):
    solver_id = SupportedQPSolver.gurobi
    STATUS_VALUE_DICT = {getattr(gurobipy.GRB.status, name): name for name in dir(gurobipy.GRB.status) if
                         '__' not in name}
    sparse = True
    compute_nI_I = False
    _times: Dict[Tuple[int, int, int], list] = defaultdict(list)

    @profile
    def init(self, H: np.ndarray, g: np.ndarray, E: np.ndarray, b: np.ndarray, A: np.ndarray, lb: np.ndarray,
             ub: np.ndarray, h: np.ndarray):
        self.qpProblem = gurobipy.Model('qp')
        self.x = self.qpProblem.addMVar(H.shape[0], lb=lb, ub=ub)
        H = sparse.diags(H, 0)
        self.qpProblem.setMObjective(Q=H, c=g, constant=0.0, xQ_L=self.x, xQ_R=self.x, sense=GRB.MINIMIZE)
        try:
            self.qpProblem.addMConstr(E, self.x, gurobipy.GRB.EQUAL, b)
        except (GurobiError, ValueError) as e:
            pass  # no eq constraints
        try:
            self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.LESS_EQUAL, h)
        except (GurobiError, ValueError) as e:
            pass # no neq constraints
        self.started = False

    def print_debug(self):
        gurobipy.setParam('LogToConsole', True)
        logging.logwarn(error_info[self.qpProblem.status])
        self.qpProblem.reset()
        self.qpProblem.optimize()
        self.qpProblem.printStats()
        self.qpProblem.printQuality()
        gurobipy.setParam('LogToConsole', False)

    def analyze_infeasibility(self):
        self.qpProblem.computeIIS()
        constraint_filter = self.qpProblem.IISConstr
        lb_filter = np.array(self.qpProblem.IISLB, dtype=bool)
        ub_filter = np.array(self.qpProblem.IISUB, dtype=bool)
        eq_constraint_ids = np.array(constraint_filter[:self.E.shape[0]], dtype=bool)
        neq_constraint_ids = constraint_filter[self.E.shape[0]:]
        num_nA_rows = np.where(self.nlbA_filter_half)[0].shape[0]
        lbA_constraint_ids = np.array(neq_constraint_ids[:num_nA_rows], dtype=bool)
        ubA_constraint_ids = np.array(neq_constraint_ids[num_nA_rows:], dtype=bool)
        return lb_filter, ub_filter, eq_constraint_ids, lbA_constraint_ids, ubA_constraint_ids

    @profile
    def problem_data_to_qp_format(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lb, ub = self.lb_ub_with_inf(self.nlb, self.ub)
        return self.weights, self.g, self.E, self.bE, self.nA_A, lb, ub, self.nlbA_ubA

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
        if success in {gurobipy.GRB.INFEASIBLE, gurobipy.GRB.INF_OR_UNBD, gurobipy.GRB.NUMERIC}:
            raise InfeasibleException(self.STATUS_VALUE_DICT[success], success)
        raise QPSolverException(self.STATUS_VALUE_DICT[success], success)

    def default_interface_solver_call(self, H, g, lb, ub, E, bE, A, lbA, ubA) -> np.ndarray:
        weights = H.diagonal()
        A = np.vstack((-A, A))
        h = np.concatenate((-lbA, ubA))
        return self.solver_call(weights, g, E, bE, A, lb, ub, h)

    def compute_violated_constraints(self, weights: np.ndarray, nA_A: np.ndarray, nlb: np.ndarray,
                                     ub: np.ndarray, nlbA_ubA: np.ndarray):
        nlb_relaxed = nlb.copy()
        ub_relaxed = ub.copy()
        if self.num_slack_variables > 0:
            lb_non_slack_without_inf = np.where(self.lb_inf_filter[:self.num_non_slack_variables])[0].shape[0]
            ub_non_slack_without_inf = np.where(self.ub_inf_filter[:self.num_non_slack_variables])[0].shape[0]
            nlb_relaxed[lb_non_slack_without_inf:] += self.retry_added_slack
            ub_relaxed[ub_non_slack_without_inf:] += self.retry_added_slack
        else:
            raise InfeasibleException('Can\'t relax constraints, because there are none.')
        # nlb_relaxed += 0.01
        # ub_relaxed += 0.01
        try:
            lb_relaxed_inf, ub_relaxed_inf = self.lb_ub_with_inf(nlb_relaxed, ub_relaxed)
            xdot_full = self.solver_call(H=self.weights, g=self.g, E=self.E, b=self.bE, A=self.nA_A,
                                         lb=lb_relaxed_inf, ub=ub_relaxed_inf, h=self.nlbA_ubA)
        except QPSolverException as e:
            self.retries_with_relaxed_constraints += 1
            raise e
        eps = 1e-4
        self.lb_filter = self.lb_inf_filter[self.weight_filter]
        self.ub_filter = self.ub_inf_filter[self.weight_filter]
        lower_violations = xdot_full[self.lb_filter] < - nlb - eps
        upper_violations = xdot_full[self.ub_filter] > ub + eps
        self.lb_filter[self.lb_filter] = lower_violations
        self.ub_filter[self.ub_filter] = upper_violations
        self.lb_filter[:self.num_non_slack_variables] = False
        self.ub_filter[:self.num_non_slack_variables] = False
        nlb_relaxed[lb_non_slack_without_inf:] -= self.retry_added_slack
        ub_relaxed[ub_non_slack_without_inf:] -= self.retry_added_slack
        nlb_relaxed[lower_violations] += self.retry_added_slack
        ub_relaxed[upper_violations] += self.retry_added_slack
        return self.lb_filter, nlb_relaxed, self.ub_filter, ub_relaxed