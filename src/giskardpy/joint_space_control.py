from collections import OrderedDict
from time import time

from giskardpy.input_system import ControllerInputArray
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint


class JointSpaceControl(QPController):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(JointSpaceControl, self).__init__(robot)

    def add_inputs(self, robot):
        self.goal_joint_states = ControllerInputArray(robot.get_joint_names(), suffix='goal')
        self.goal_weights = ControllerInputArray(robot.get_joint_names(), suffix='sc_w')

    def make_constraints(self, robot):
        t = time()
        default_weights = OrderedDict()
        for joint_name in robot.get_joint_names():
            joint_symbol = robot.get_joint_state_input().to_symbol(joint_name)
            goal = self.goal_joint_states.to_symbol(joint_name)
            weight = self.goal_weights.to_symbol(joint_name)
            sc = SoftConstraint(lower=goal - joint_symbol, upper=goal - joint_symbol,
                                weight=weight, expression=joint_symbol)
            self._soft_constraints[joint_name] = sc
            default_weights[joint_name] = self.weight

        self.update_observables(self.goal_weights.get_update_dict(**default_weights))
        super(JointSpaceControl, self).make_constraints(robot)
        print('make constraints took {}'.format(time() - t))

    def set_goal(self, goal_joint_state):
        self.update_observables(self.goal_joint_states.get_update_dict(**goal_joint_state))
