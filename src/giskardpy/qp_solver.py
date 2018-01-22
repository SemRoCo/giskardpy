import numpy as np
import qpoases
from qpoases import PyReturnValue


class QPSolver(object):
    RETURN_VALUE_DICT = {value: name for name, value in vars(PyReturnValue).iteritems()}

    def __init__(self, dim_a, dim_b):
        self.qpProblem = qpoases.PySQProblem(dim_a, dim_b)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        self.qpProblem.setOptions(options)
        self.xdot_full = np.zeros(dim_a)

        self.started = False

    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
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
