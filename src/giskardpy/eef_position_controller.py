from giskardpy.controller import Controller
from giskardpy.sympy_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint

class EEFPositionControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        self.goal = {}
        super(EEFPositionControl, self).__init__(robot)

    def make_constraints(self, robot):
        self.goal_expr = inputPoint3('goal_pos', self._observables)
        dist = norm(posOf(robot.eef) - self.goal_expr)
        print(robot.eef)
        self.soft_constraints['align eef position'] = SoftConstraint(lower=-dist,
                                                                     upper=-dist,
                                                                     weight=self.weight,
                                                                     expression=dist)

    def set_goal(self, goal_pos):
        self.goal['goal_pos'] = goal_pos
        expandVec3Input('goal_pos', self.goal)

    def update_observables(self, updates=None):
        return super(EEFPositionControl, self).update_observables(self.goal)