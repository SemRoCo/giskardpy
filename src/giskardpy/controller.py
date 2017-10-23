from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot
from giskardpy.sympy_wrappers import *


class Controller(object):
    def __init__(self, robot):
        self.robot = robot

        self.__state = OrderedDict()  # e.g. goal
        self._controllable_constraints = OrderedDict()
        self._hard_constraints = OrderedDict()
        self._soft_constraints = OrderedDict()

        self.add_inputs(robot)
        self.make_constraints(self.robot)
        self.build_builder()

    def get_state(self):
        return self.__state

    def add_inputs(self, robot):
        pass

    def make_constraints(self, robot):
        self._controllable_constraints = self.robot.joint_constraints
        self._hard_constraints = self.robot.hard_constraints

    def build_builder(self):
        self.qp_problem_builder = QProblemBuilder(self._controllable_constraints,
                                                  self._hard_constraints,
                                                  self._soft_constraints)

    def update_observables(self, updates):
        """
        :param updates: dict{str->float} observable name to it value
        :return: dict{str->float} joint name to vel command
        """
        self.__state.update(updates)

    def get_next_command(self):
        self.__state.update(self.robot.get_state())
        return self.qp_problem_builder.update_observables(self.__state)
