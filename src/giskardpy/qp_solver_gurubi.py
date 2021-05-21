import numpy as np
import gurobipy

from giskardpy.exceptions import QPSolverException, InfeasibleException
from giskardpy import logging
from giskardpy.qp_solver import QPSolver

gurobipy.setParam('LogToConsole', False)

class QPSolverGurubi(QPSolver):
    STATUS_VALUE_DICT = {getattr(gurobipy.GRB.status, name): name for name in dir(gurobipy.GRB.status) if '__' not in name}

    def __init__(self):
        """
        :param dim_a: number of joint constraints + number of soft constraints
        :type int
        :param dim_b: number of hard constraints + number of soft constraints
        :type int
        """
        self.started = False
        self.shape = (0,0)

    @profile
    def init(self, H, g, A, lb, ub, lbA, ubA):
        self.qpProblem = gurobipy.Model('qp')
        x = self.qpProblem.addMVar(lb.shape, lb=lb, ub=ub)
        self.qpProblem.addMConstr(A, x, gurobipy.GRB.LESS_EQUAL, ubA)
        self.qpProblem.addMConstr(A, x, gurobipy.GRB.GREATER_EQUAL, lbA)
        self.qpProblem.setMObjective(H, None, 0.0)
        self.started = False
        return True

    def print_debug(self):
        gurobipy.setParam('LogToConsole', True)
        self.qpProblem.reset()
        self.qpProblem.optimize()
        self.qpProblem.printStats()
        self.qpProblem.printQuality()
        gurobipy.setParam('LogToConsole', False)

    @profile
    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
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
        if A.shape != self.shape:
            self.started = False
            self.shape = A.shape

        number_of_retries = 2
        r = 4
        while number_of_retries > 0:
            number_of_retries -= 1
            self.init(H, g, A, lb, ub, lbA, ubA)
            self.qpProblem.optimize()
            success = self.qpProblem.status
            if success in [gurobipy.GRB.OPTIMAL, gurobipy.GRB.SUBOPTIMAL]:
                self.xdot_full = np.array([x.x for x in self.qpProblem.getVars()])
                self.started = True
                break
            else:
                logging.loginfo(u'optimization unsuccessful {}'.format(self.STATUS_VALUE_DICT[success]))
                self.print_debug()
                logging.loginfo(u'retrying with A rounded to {} decimal places'.format(r))
                # A = np.round(A, r)
                # lbA = np.round(lbA, r)
                # ubA = np.round(ubA, r)
                r -= 1
                self.started = False
        else:  # if not break
            self.started = False
            self.print_debug()
            raise QPSolverException(u'{}'.format(self.STATUS_VALUE_DICT[success]))

        return self.xdot_full
