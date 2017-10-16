from giskardpy.qp_problem_builder import QProblemBuilder
import numpy as np
import qpoases
from qpoases import PyReturnValue


class QPController(object):
    RETURN_VALUE_DICT = dict([(value, name) for name, value in vars(PyReturnValue).iteritems()])

    def __init__(self, controller):
        self.controller = controller
        self.qpBuilder = QProblemBuilder(self.controller)

    def start(self, nWSR):
        self.qpBuilder.update()
        self.qpProblem = qpoases.PySQProblem(*self.qpBuilder.get_problem_dimensions())
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        self.qpProblem.setOptions(options)
        success = self.qpProblem.init(self.qpBuilder.get_H(), self.qpBuilder.get_g(), self.qpBuilder.get_A(),
                                      self.qpBuilder.get_lb(), self.qpBuilder.get_ub(),
                                      self.qpBuilder.get_lbA(), self.qpBuilder.get_ubA(), nWSR)

        if success != PyReturnValue.SUCCESSFUL_RETURN:
            print("Failed to initialize QP-problem. ERROR: {}".format(self.RETURN_VALUE_DICT[success]))
            return False

        self.xdot_full = np.zeros(self.qpBuilder.get_num_controllables() + self.qpBuilder.get_num_soft_constraints())
        self.xdot_control = np.zeros(self.qpBuilder.get_num_controllables())
        self.xdot_slack = np.zeros(self.qpBuilder.get_num_soft_constraints())

        return True

    def update(self, nWSR):
        self.qpBuilder.update()
        if self.xdot_control is None:
            raise Exception("Attempted to update controller without starting it first.")

        success = self.qpProblem.hotstart(self.qpBuilder.get_H(), self.qpBuilder.get_g(), self.qpBuilder.get_A(),
                                          self.qpBuilder.get_lb(), self.qpBuilder.get_ub(),
                                          self.qpBuilder.get_lbA(), self.qpBuilder.get_ubA(), nWSR)

        if success != PyReturnValue.SUCCESSFUL_RETURN:
            print("Failed to initialize QP-problem. ERROR: {}".format(self.RETURN_VALUE_DICT[success]))
            return False

        self.qpProblem.getPrimalSolution(self.xdot_full)
        self.xdot_control = self.xdot_full[:self.qpBuilder.get_num_controllables()]
        self.xdot_slack = self.xdot_full[self.qpBuilder.get_num_controllables():]

        return True

    def get_command_vector(self):
        return self.xdot_control
