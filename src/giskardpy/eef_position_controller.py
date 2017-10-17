from giskardpy.controller import Controller
from giskardpy.sympy_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint

class EEFPositionControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        self.goal = {}
        super(EEFPositionControl, self).__init__(robot)

    def make_constraints(self, robot):
        self.goal_expr = inputVec3('goal_pos', self._observables)
        dist = (robot.eef.position_wrt(odom) - self.goal_expr).magnitude()
        print(robot.eef.position_wrt(odom))
        self.soft_constraints['align eef position'] = SoftConstraint(lower=-dist,
                                                                     upper=-dist,
                                                                     weight=self.weight,
                                                                     expression=dist)

    def set_goal(self, goal_pos):
        self.goal['goal_pos'] = goal_pos
        expandVec3Input('goal_pos', self.goal)

    def update_observables(self, updates=None):
        return super(EEFPositionControl, self).update_observables(self.goal)