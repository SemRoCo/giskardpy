from giskardpy.qpcontroller import QPController
from giskardpy.sympy_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput


class EEFPositionControl(QPController):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(EEFPositionControl, self).__init__(robot)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.goal_weights = {}
        for eef in robot.end_effectors:
            self.goal_eef[eef] = Point3Input(prefix=eef, suffix='goal')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        for eef in robot.end_effectors:
            eef_frame = robot.frames[eef]
            self.goal_expr = self.goal_eef[eef].get_expression()
            dist = norm(pos_of(eef_frame) - self.goal_expr)
            self._soft_constraints['align {} position'.format(eef)] = SoftConstraint(lower=-dist,
                                                                         upper=-dist,
                                                                         weight=self.goal_weights[eef].get_expression(),
                                                                         expression=dist)
            self._controllable_constraints = robot.joint_constraints
            self._hard_constraints = robot.hard_constraints
            self.update_observables({self.goal_weights[eef].get_symbol_str(): self.weight})
            self.set_goal({eef: robot.get_eef_position()[eef][:3, 3]})

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        for eef, goal_pos in goal.items():
            self.update_observables(self.goal_eef[eef].get_update_dict(*goal_pos))
