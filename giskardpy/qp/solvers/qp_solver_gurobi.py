from __future__ import annotations
from collections import defaultdict
from typing import Tuple, Dict, TYPE_CHECKING

import gurobipy
import numpy as np
from giskardpy.qp.adapters.qp_adapter import QPData, GiskardToExplicitQPAdapter
from gurobipy import GRB, GurobiError
from line_profiler import profile
from giskardpy.qp.exceptions import QPSolverException, InfeasibleException
from giskardpy.middleware import get_middleware
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.qp.solvers.qp_solver_ids import SupportedQPSolver

if TYPE_CHECKING:
    import scipy.sparse as sp

gurobipy.setParam(gurobipy.GRB.Param.LogToConsole, False)
gurobipy.setParam(gurobipy.GRB.Param.FeasibilityTol, 2.5e-5)

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
    """
    min_x 0.5 x^T P x + q^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """

    solver_id = SupportedQPSolver.gurobi
    required_adapter_type = GiskardToExplicitQPAdapter
    STATUS_VALUE_DICT = {
        getattr(gurobipy.GRB.status, name): name
        for name in dir(gurobipy.GRB.status)
        if "__" not in name
    }
    _times: Dict[Tuple[int, int, int], list] = defaultdict(list)

    @profile
    def init(self, qp_data: QPData):
        import scipy.sparse as sp

        self.qpProblem = gurobipy.Model("qp")
        self.x = self.qpProblem.addMVar(
            qp_data.quadratic_weights.shape[0],
            lb=qp_data.box_lower_constraints,
            ub=qp_data.box_upper_constraints,
        )
        H = sp.diags(qp_data.quadratic_weights, 0)
        self.qpProblem.setMObjective(
            Q=H,
            c=qp_data.linear_weights,
            constant=0.0,
            xQ_L=self.x,
            xQ_R=self.x,
            sense=GRB.MINIMIZE,
        )
        try:
            self.qpProblem.addMConstr(
                qp_data.eq_matrix, self.x, gurobipy.GRB.EQUAL, qp_data.eq_bounds
            )
        except (GurobiError, ValueError) as e:
            pass  # no eq constraints
        try:
            self.qpProblem.addMConstr(
                qp_data.neq_matrix,
                self.x,
                gurobipy.GRB.GREATER_EQUAL,
                qp_data.neq_lower_bounds,
            )
            self.qpProblem.addMConstr(
                qp_data.neq_matrix,
                self.x,
                gurobipy.GRB.LESS_EQUAL,
                qp_data.neq_upper_bounds,
            )
        except (GurobiError, ValueError) as e:
            pass  # no neq constraints
        self.started = False

    def print_debug(self):
        gurobipy.setParam(gurobipy.GRB.Param.LogToConsole, True)
        get_middleware().logwarn(error_info[self.qpProblem.status])
        self.qpProblem.reset()
        self.qpProblem.optimize()
        self.qpProblem.printStats()
        self.qpProblem.printQuality()
        gurobipy.setParam(gurobipy.GRB.Param.LogToConsole, False)

    def analyze_infeasibility(self):
        self.qpProblem.computeIIS()
        constraint_filter = self.qpProblem.IISConstr
        lb_filter = np.array(self.qpProblem.IISLB, dtype=bool)
        ub_filter = np.array(self.qpProblem.IISUB, dtype=bool)
        eq_constraint_ids = np.array(constraint_filter[: self.E.shape[0]], dtype=bool)
        neq_constraint_ids = constraint_filter[self.E.shape[0] :]
        num_nA_rows = np.where(self.nlbA_filter_half)[0].shape[0]
        lbA_constraint_ids = np.array(neq_constraint_ids[:num_nA_rows], dtype=bool)
        ubA_constraint_ids = np.array(neq_constraint_ids[num_nA_rows:], dtype=bool)
        return (
            lb_filter,
            ub_filter,
            eq_constraint_ids,
            lbA_constraint_ids,
            ubA_constraint_ids,
        )

    def solver_call_explicit_interface(self, qp_data: QPData) -> np.ndarray:
        self.init(qp_data)
        self.qpProblem.optimize()
        success = self.qpProblem.status
        if success in {gurobipy.GRB.OPTIMAL, gurobipy.GRB.SUBOPTIMAL}:
            if success == gurobipy.GRB.SUBOPTIMAL:
                get_middleware().logwarn("warning, suboptimal solution!")
            return np.array(self.qpProblem.X)
        if success in {
            gurobipy.GRB.INFEASIBLE,
            gurobipy.GRB.INF_OR_UNBD,
            gurobipy.GRB.NUMERIC,
        }:
            raise InfeasibleException(self.STATUS_VALUE_DICT[success])
        raise QPSolverException(self.STATUS_VALUE_DICT[success])

    solver_call = solver_call_explicit_interface
