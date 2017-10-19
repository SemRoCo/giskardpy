from copy import deepcopy

from giskardpy.controller import Controller
import sympy as sp
import numpy as np

from giskardpy.qp_problem_builder import SoftConstraint


class JointSpaceControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(JointSpaceControl, self).__init__(robot)

    def make_constraints(self, robot):
        for i, joint_symbol in enumerate(robot.joints_observables):
            goal = sp.Symbol('{}_goal'.format(joint_symbol))
            self._observables.append(goal)
            self._soft_constraints['soft_{}'.format(i)] = SoftConstraint(lower=goal - joint_symbol,
                                                                         upper=goal - joint_symbol,
                                                                         weight=self.weight,
                                                                         expression=joint_symbol)

    def set_goal(self, goal_dict):
        self.goal = goal_dict

    def update_observables(self, updates=None):
        return super(JointSpaceControl, self).update_observables(self.goal)
