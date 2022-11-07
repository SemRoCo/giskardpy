import numpy as np
import qpoases
from qpoases import PyReturnValue

from giskardpy.exceptions import MAX_NWSR_REACHEDException, QPSolverException, InfeasibleException
from giskardpy.qp.qp_solver import QPSolver
from giskardpy.utils import logging


class QPSolverQPOases(QPSolver):
    STATUS_VALUE_DICT = {value: name for name, value in vars(PyReturnValue).items()}

    def __init__(self):
        """
        :param dim_a: number of joint constraints + number of soft constraints
        :type int
        :param dim_b: number of hard constraints + number of soft constraints
        :type int
        """
        self.started = False
        self.default = False
        self.shape = (0,0)

    def init(self, dim_a, dim_b):
        self.qpProblem = qpoases.PySQProblem(dim_a, dim_b)
        options = qpoases.PyOptions()
        if self.default:
            options.setToDefault()
        else:
            options.setToMPC()
        options.printLevel = qpoases.PyPrintLevel.NONE
        self.qpProblem.setOptions(options)
        self.xdot_full = np.zeros(dim_a)

        self.started = False

    # @profile
    def solve(self, H, g, A, lb, ub, lbA, ubA, nWSR=None):
        H = np.diag(H)
        H = H.copy()
        A = A.copy()
        lbA = lbA.copy()
        ubA = ubA.copy()
        lb = lb.copy()
        ub = ub.copy()
        if A.shape != self.shape:
            self.started = False
            self.default = False
            self.shape = A.shape

        number_of_retries = 2
        while number_of_retries > 0:
            if nWSR is None:
                nWSR = np.array([sum(A.shape) * 2])
            else:
                nWSR = np.array([nWSR])
            number_of_retries -= 1
            if not self.started:
                self.init(A.shape[1], A.shape[0])
                success = self.qpProblem.init(H, g, A, lb, ub, lbA, ubA, nWSR)
                if success == PyReturnValue.MAX_NWSR_REACHED:
                    self.started = False
                    raise MAX_NWSR_REACHEDException('Failed to initialize QP-problem.')
            else:
                success = self.qpProblem.hotstart(H, g, A, lb, ub, lbA, ubA, nWSR)
                if success == PyReturnValue.MAX_NWSR_REACHED:
                    logging.logwarn('max nwsr or cpu time reached')
                    success = PyReturnValue.SUCCESSFUL_RETURN
                    # self.started = False
                    # raise MAX_NWSR_REACHEDException('Failed to hot start QP-problem.')
            if success == PyReturnValue.SUCCESSFUL_RETURN:
                self.started = True
                break
            elif success == PyReturnValue.NAN_IN_LB:
                # TODO nans get replaced with 0 document this somewhere
                # TODO might still be buggy when nan occur when the qp problem is already initialized
                lb[np.isnan(lb)] = 0
                nWSR = None
                self.started = False
                number_of_retries += 1
                continue
            elif success == PyReturnValue.NAN_IN_UB:
                ub[np.isnan(ub)] = 0
                nWSR = None
                self.started = False
                number_of_retries += 1
                continue
            elif success == PyReturnValue.NAN_IN_LBA:
                lbA[np.isnan(lbA)] = 0
                nWSR = None
                self.started = False
                number_of_retries += 1
                continue
            elif success == PyReturnValue.NAN_IN_UBA:
                ubA[np.isnan(ubA)] = 0
                nWSR = None
                self.started = False
                number_of_retries += 1
                continue
            elif number_of_retries == 1:
                logging.loginfo(f'{self.STATUS_VALUE_DICT[success]}; retrying with A rounded to 5 decimal places')
                decimal_places = 5
                H = np.round(H, decimal_places)
                A = np.round(A, decimal_places)
                lb = np.round(lb, decimal_places)
                ub = np.round(ub, decimal_places)
                lbA = np.round(lbA, decimal_places)
                ubA = np.round(ubA, decimal_places)
                nWSR = None
                self.started = False
            else:
                if not self.default:
                    logging.loginfo(f'{self.STATUS_VALUE_DICT[success]}; retrying with default mode.')
                    self.default = True
                    self.started = False
        else:  # if not break
            self.started = False
            message = '{}'.format(self.STATUS_VALUE_DICT[success])
            if success in [PyReturnValue.INIT_FAILED_INFEASIBILITY,
                           PyReturnValue.QP_INFEASIBLE,
                           PyReturnValue.HOTSTART_STOPPED_INFEASIBILITY,
                           PyReturnValue.ADDBOUND_FAILED_INFEASIBILITY,
                           PyReturnValue.ADDCONSTRAINT_FAILED_INFEASIBILITY]:
                raise InfeasibleException(message)
            raise QPSolverException(message)

        self.qpProblem.getPrimalSolution(self.xdot_full)
        return self.xdot_full
