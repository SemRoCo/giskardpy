import numpy as np
import gurobipy

from giskardpy.exceptions import QPSolverException, InfeasibleException
from giskardpy import logging
from giskardpy.qp_solver import QPSolver

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

class QPSolverGurubi(QPSolver):
    STATUS_VALUE_DICT = {getattr(gurobipy.GRB.status, name): name for name in dir(gurobipy.GRB.status) if '__' not in name}

    def __init__(self):
        self.started = False

    @profile
    def init(self, H, g, A, lb, ub, lbA, ubA):
        # TODO potential speed up by reusing model
        self.qpProblem = gurobipy.Model('qp')
        self.x = self.qpProblem.addMVar(lb.shape, lb=lb, ub=ub)
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.LESS_EQUAL, ubA)
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.GREATER_EQUAL, lbA)
        self.qpProblem.setMObjective(H, None, 0.0)
        self.started = True

    @profile
    def update(self, H, g, A, lb, ub, lbA, ubA):
        # self.init(H, g, A, lb, ub, lbA, ubA)
        # return
        self.x.lb = lb
        self.x.ub = ub
        self.qpProblem.remove(self.qpProblem.getConstrs())
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.LESS_EQUAL, ubA)
        self.qpProblem.addMConstr(A, self.x, gurobipy.GRB.GREATER_EQUAL, lbA)
        self.qpProblem.setMObjective(H, None, 0.0)
        pass

    def print_debug(self):
        # TODO use MinRHS etc to analyse solution
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
    def solve(self, H, g, A, lb, ub, lbA, ubA, tries=1, decimal_places=4):
        """
        x^T*H*x + x^T*g
        s.t.: lbA < A*x < ubA
        and    lb <  x  < ub
        :param H: 2d diagonal weight matrix, shape = (jc (joint constraints) + sc (soft constraints)) * (jc + sc)
        :type np.array
        :param g: 1d zero vector of len joint constraints + soft constraints
        :type np.array
        :param A: 2d jacobi matrix of hc (hard constraints) and sc, shape = (hc + sc) * (number of joints)
        :type np.array
        :param lb: 1d vector containing lower bound of x, len = jc + sc
        :type np.array
        :param ub: 1d vector containing upper bound of x, len = js + sc
        :type np.array
        :param lbA: 1d vector containing lower bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param ubA: 1d vector containing upper bounds for the change of hc and sc, len = hc+sc
        :type np.array
        :param nWSR:
        :type np.array
        :return: x according to the equations above, len = joint constraints + soft constraints
        :type np.array
        """
        for i in range(tries):
            if self.started:
                self.update(H, g, A, lb, ub, lbA, ubA)
            else:
                self.init(H, g, A, lb, ub, lbA, ubA)
            self.qpProblem.optimize()
            success = self.qpProblem.status
            if success in {gurobipy.GRB.OPTIMAL, gurobipy.GRB.SUBOPTIMAL}:
                if success == gurobipy.GRB.SUBOPTIMAL:
                    logging.logwarn('warning, suboptimal!')
                self.xdot_full = np.array(self.qpProblem.X)
                break
            elif success in {gurobipy.GRB.NUMERIC} and i < tries-1:
                self.print_debug()
                logging.logwarn(u'Solver returned \'{}\', retrying with data rounded to \'{}\' decimal places'.format(
                    self.STATUS_VALUE_DICT[success],
                    decimal_places
                ))
                H = self.round(H,decimal_places)
                A = self.round(A,decimal_places)
                lb = self.round(lb,decimal_places)
                ub = self.round(ub,decimal_places)
                lbA = self.round(lbA,decimal_places)
                ubA = self.round(ubA,decimal_places)
        else:
            self.print_debug()
            self.started = False
            error_message = u'{}'.format(self.STATUS_VALUE_DICT[success])
            if success == gurobipy.GRB.INFEASIBLE:
                raise InfeasibleException(error_message)
            raise QPSolverException(error_message)

        return self.xdot_full
