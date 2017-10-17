from copy import deepcopy

from giskardpy.controller import Controller
import sympy as sp
import numpy as np


class JointSpaceControl(Controller):
    def __init__(self, robot, weights=42):
        super(JointSpaceControl, self).__init__(robot)
        self._weights = np.ones(3) * weights

        for joint_symbol in robot.joints_observables:
            goal = sp.Symbol('{}_goal'.format(joint_symbol))
            self._observables.append(goal)
            self._soft_expressions.append(goal)
            self._lbA.append(goal - joint_symbol)
            self._ubA.append(goal - joint_symbol)
            self._lb.append(-10e6)
            self._ub.append(10e6)

    def set_goal(self, goal_dict):
        self.goal = goal_dict

    def update_observables(self):
        updates = deepcopy(self.robot.update_observables())
        updates.update(self.goal)
        return updates
