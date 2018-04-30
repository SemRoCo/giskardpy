import numpy as np
import qpoases
from qpoases import PyReturnValue


class QPSolver(object):
    RETURN_VALUE_DICT = {value: name for name, value in vars(PyReturnValue).items()}

    def __init__(self, dim_a, dim_b):
        """
        :param dim_a: number of joint constraints + number of soft constraints
        :type int
        :param dim_b: number of hard constraints + number of soft constraints
        :type int
        """
        self.qpProblem = qpoases.PySQProblem(dim_a, dim_b)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        self.qpProblem.setOptions(options)
        self.xdot_full = np.zeros(dim_a)

        self.started = False

    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        """
        x^T*H*x
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
        :return: x according to the equations above, len = number of joints
        :type np.array
        """
        if nWSR is None:
            nWSR = np.array([100])
        if not self.started:
            success = self.qpProblem.init(H, g, A, lb, ub, lbA, ubA, nWSR)
            if success != PyReturnValue.SUCCESSFUL_RETURN:
                print("Failed to initialize QP-problem. ERROR: {}".format(self.RETURN_VALUE_DICT[success]))
                return None
            self.started = True
        else:
            success = self.qpProblem.hotstart(H, g, A, lb, ub, lbA, ubA, nWSR)
            if success != PyReturnValue.SUCCESSFUL_RETURN:
                print("Failed to hot start QP-problem. ERROR: {}".format(self.RETURN_VALUE_DICT[success]))
                self.started = False
                return None

        self.qpProblem.getPrimalSolution(self.xdot_full)
        return self.xdot_full