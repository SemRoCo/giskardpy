from collections import namedtuple

import sympy as sp
import numpy as np

from giskardpy.qp_problem_builder import QProblemBuilder
from giskardpy.robot import Robot



class Controller(object):
    def __init__(self, robot):
        # TODO: replace
        self.robot = robot
        # self.robot = Robot()


        #TODO: fill in child class
        self._observables = []
        self._weights = []
        self._soft_expressions = []
        self._lb = []
        self._ub = []
        self._lbA = []  # soft lb
        self._ubA = []  # soft ub

        # self.qp_problem_builder = QProblemBuilder()

    def set_goal(self, goal_dict):
        pass

    def update_observables(self):
        updates = {}
        robot_updates = self.robot.update_observables()
        updates.update(robot_updates)
        return updates

    def get_hard_expressions(self):
        return self.robot.hard_expressions

    def get_soft_expressions(self):
        return self._soft_expressions

    def get_weights(self):
        return np.concatenate((self.robot.weights, self._weights))

    def get_num_controllables(self):
        return len(self.robot.lb)

    def get_observables(self):
        return self.get_robot_observables() + self.get_controller_observables()

    def get_controller_observables(self):
        return self._observables

    def get_robot_observables(self):
        return self.robot.observables

    def get_lb(self):
        return self.robot.lb + self._lb

    def get_ub(self):
        return self.robot.ub + self._ub

    def get_lbA(self):
        return self.robot.lbA + self._lbA

    def get_ubA(self):
        return self.robot.ubA + self._ubA
