from time import time

from giskardpy import USE_SYMENGINE
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput, FrameInput

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw


class CartesianControllerOld(QPController):
    def __init__(self, robot, builder_backend=None, weight=1, gain=20, threshold_value=.01):
        self.weight = weight
        self.default_gain = gain
        self.default_threshold = threshold_value
        super(CartesianControllerOld, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.goal_weights = {}
        self.gain = ScalarInput(prefix='gain')
        self.threshold_value = ScalarInput(prefix='threshold_value')
        for eef in robot.end_effectors:
            self.goal_eef[eef] = FrameInput(prefix=eef, suffix='goal')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        t = time()
        for eef in robot.end_effectors:
            start_pose = robot.get_eef_position2()[eef]
            # start_position = spw.pos_of(start_pose)

            current_pose = robot.frames[eef]
            current_position = spw.pos_of(current_pose)
            current_rotation = spw.rot_of(current_pose)[:3, :3]

            goal_position = self.goal_eef[eef].get_position()
            goal_rotation = self.goal_eef[eef].get_rotation()[:3, :3]
            # goal_r = goal_rotation[:3,:3].reshape(9,1)

            # pos control ----------------------------------------------------------------------------------------------
            dist = spw.norm(current_position - goal_position)
            dist_control = spw.Min(dist, self.threshold_value.get_expression())  * self.gain.get_expression()

            # rot control ----------------------------------------------------------------------------------------------
            # dist_r = spw.rotation_distance(current_rotation, goal_rotation)
            dist_r = spw.norm(current_rotation.reshape(9, 1) - goal_rotation.reshape(9, 1))
            # dist_r_control = spw.Min(dist_r, self.threshold_value.get_expression())
            dist_r_control = spw.Min(dist_r, 0.2)
            dist_r_control = dist_r_control * self.gain.get_expression()/3.

            self._soft_constraints['align {} position'.format(eef)] = SoftConstraint(lower=-dist_control,
                                                                                     upper=-dist_control,
                                                                                     weight=self.goal_weights[
                                                                                         eef].get_expression(),
                                                                                     expression=dist)
            self._soft_constraints['align {} rotation'.format(eef)] = SoftConstraint(lower=-dist_r_control,
                                                                                     upper=-dist_r_control,
                                                                                     weight=self.goal_weights[
                                                                                         eef].get_expression(),
                                                                                     expression=dist_r)
            self._controllable_constraints = robot.joint_constraints
            self._hard_constraints = robot.hard_constraints
            self.update_observables({self.goal_weights[eef].get_symbol_str(): self.weight})
            self.set_goal({eef: start_pose})
        self.update_observables({self.gain.get_symbol_str(): self.default_gain})
        self.update_observables({self.threshold_value.get_symbol_str(): self.default_threshold})
        print('make constraints took {}'.format(time() - t))

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        for eef, goal_pos in goal.items():
            self.update_observables(self.goal_eef[eef].get_update_dict(*goal_pos))
