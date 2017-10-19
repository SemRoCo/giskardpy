from collections import OrderedDict

import sympy as sp
import numpy as np

from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot



class Controller(object):
    def __init__(self, robot):
        self.robot = robot

        self._observables = []
        self._soft_constraints = OrderedDict()

        self.build_builder()

    def make_constraints(self, robot):
        pass

    def build_builder(self):
        self.make_constraints(self.robot)

        self.qp_problem_builder = QProblemBuilder(self.robot.joint_constraints,
                                                  self.robot.hard_constraints,
                                                  self._soft_constraints)

    def set_goal(self, goal_dict):
        pass

    def update_observables(self, updates=None):
        """
        :param updates: dict{str->float} observable name to it value
        :return: dict{str->float} joint name to vel command
        """
        if updates is None:
            updates = {}
        robot_updates = self.robot.update_observables()
        updates.update(robot_updates)
        return self.qp_problem_builder.update_observables(updates)


    def get_controller_observables(self):
        return self._observables