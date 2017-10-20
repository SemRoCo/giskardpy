from giskardpy.controller import Controller
import sympy as sp

from giskardpy.qp_problem_builder import SoftConstraint


class JointSpaceControl(Controller):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(JointSpaceControl, self).__init__(robot)

    def make_constraints(self, robot):
        for i, joint_name in enumerate(robot.get_joint_names()):
            joint_symbol = sp.Symbol(joint_name)
            goal_name = '{}_goal'.format(joint_symbol)
            goal = sp.Symbol(goal_name)
            self._state[goal_name] = None
            self._soft_constraints['soft_{}'.format(i)] = SoftConstraint(lower=goal - joint_symbol,
                                                                         upper=goal - joint_symbol,
                                                                         weight=self.weight,
                                                                         expression=joint_symbol)

    def set_goal(self, goal_dict):
        self.update_observables(goal_dict)
