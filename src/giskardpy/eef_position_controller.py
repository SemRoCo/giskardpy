from giskardpy.controller import Controller
from giskardpy.sympy_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input

class EEFPositionControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(EEFPositionControl, self).__init__(robot)

    def make_constraints(self, robot):
        self.goal_input = Point3Input('eef', 'goal')
        self.goal_expr = self.goal_input.get_expression()
        dist = norm(pos_of(robot.eef) - self.goal_expr)
        print(robot.eef)
        self._soft_constraints['align eef position'] = SoftConstraint(lower=-dist,
                                                                     upper=-dist,
                                                                     weight=self.weight,
                                                                     expression=dist)
        self._controllable_constraints = robot.joint_constraints
        self._hard_constraints = robot.hard_constraints

    def set_goal(self, goal_pos):
        self.update_observables(self.goal_input.get_update_dict(*goal_pos))
