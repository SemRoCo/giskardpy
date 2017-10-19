from giskardpy.controller import Controller
from giskardpy.sympy_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint

class EEFPositionControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        self.goal = {}
        super(EEFPositionControl, self).__init__(robot)

    def make_constraints(self, robot):
        self.goal_expr = self.add_point3_input('goal_pos')
        dist = norm(pos_of(robot.eef) - self.goal_expr)
        print(robot.eef)
        self.soft_constraints['align eef position'] = SoftConstraint(lower=-dist,
                                                                     upper=-dist,
                                                                     weight=self.weight,
                                                                     expression=dist)
        self.controllable_constraints = robot.joint_constraints

    def set_goal(self, goal_pos):
        self.update_input('goal_pos', *goal_pos)