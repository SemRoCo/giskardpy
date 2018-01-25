from time import time

from giskardpy import USE_SYMENGINE
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput, FrameInput
import numpy as np

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw


class CartesianLineController(QPController):
    def __init__(self, robot, builder_backend=None, weight=1):
        self.weight = weight
        self.max_trans_speed = 0.3  # m/s
        # self.default_trans_threshold = 0
        self.default_trans_gain = 3

        # self.default_rot_threshold = 0.2
        self.max_rot_speed = 0.5  # rad/s
        self.default_rot_gain = 3
        super(CartesianLineController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.start_eef = {}
        self.goal_weights = {}
        for eef in robot.end_effectors:
            self.goal_eef[eef] = FrameInput(prefix=eef, suffix='goal')
            self.start_eef[eef] = FrameInput(prefix=eef, suffix='start')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        t = time()
        for eef in robot.end_effectors:
            start_position = self.start_eef[eef].get_position()
            start_rotation = self.start_eef[eef].get_rotation()

            current_pose = robot.frames[eef]
            current_position = spw.pos_of(current_pose)
            current_rotation = spw.rot_of(current_pose)
            current_quaternion = robot.q_rot[eef]

            goal_position = self.goal_eef[eef].get_position()
            goal_rotation = self.goal_eef[eef].get_rotation()
            goal_quaternion = self.goal_eef[eef].get_quaternion()

            # pos control ----------------------------------------------------------------------------------------------
            trans_error_vector = goal_position - current_position
            trans_error = spw.norm(trans_error_vector)
            trans_scale = spw.fake_Min(trans_error * self.default_trans_gain, self.max_trans_speed)
            trans_control = trans_error_vector / trans_error * trans_scale


            self._soft_constraints['align {} x position'.format(eef)] = SoftConstraint(lower=trans_control[0],
                                                                                       upper=trans_control[0],
                                                                                       weight=self.goal_weights[
                                                                                           eef].get_expression(),
                                                                                       expression=current_position[0])
            self._soft_constraints['align {} y position'.format(eef)] = SoftConstraint(lower=trans_control[1],
                                                                                       upper=trans_control[1],
                                                                                       weight=self.goal_weights[
                                                                                           eef].get_expression(),
                                                                                       expression=current_position[1])
            self._soft_constraints['align {} z position'.format(eef)] = SoftConstraint(lower=trans_control[2],
                                                                                       upper=trans_control[2],
                                                                                       weight=self.goal_weights[
                                                                                           eef].get_expression(),
                                                                                       expression=current_position[2])

            #line ------------------------------------------------------------------------------------------------------
            x0 = current_position[:-1, :]
            x1 = spw.Matrix([x[0] for x in (start_position[:-1, :] + (np.random.random((3, 1)) * 0.005))])
            x2 = goal_position[:-1, :]
            dist_to_line = spw.norm(spw.cross((x0 - x1), (x0 - x2))) / spw.norm(x2 - x1)
            dist_to_line = dist_to_line
            self._soft_constraints['{} stay on line'.format(eef)] = SoftConstraint(lower=-dist_to_line,
                                                                                   upper=-dist_to_line,
                                                                                   weight=self.goal_weights[
                                                                                              eef].get_expression() * 100,
                                                                                   expression=dist_to_line)

            # rot control ----------------------------------------------------------------------------------------------

            axis, angle = spw.axis_angle_from_matrix((current_rotation.T * goal_rotation))
            capped_angle = spw.fake_Min(angle * self.default_rot_gain, self.max_rot_speed)
            r_rot_control = axis * capped_angle

            hack = spw.rotation3_axis_angle([0, 0, 1], 0.0001)

            axis, angle = spw.axis_angle_from_matrix((current_rotation.T * (start_rotation * hack)).T)
            # axis, angle = spw.axis_angle_from_matrix((current_rotation.T * current_rotation).T)
            c_aa = (axis * angle)

            self._soft_constraints['align {} rotation 0'.format(eef)] = SoftConstraint(
                lower=r_rot_control[0],
                upper=r_rot_control[0],
                weight=self.goal_weights[eef].get_expression() + self.get_robot().default_joint_weight,
                expression=c_aa[0])
            self._soft_constraints['align {} rotation 1'.format(eef)] = SoftConstraint(
                lower=r_rot_control[1],
                upper=r_rot_control[1],
                weight=self.goal_weights[eef].get_expression() + self.get_robot().default_joint_weight,
                expression=c_aa[1])
            self._soft_constraints['align {} rotation 2'.format(eef)] = SoftConstraint(
                lower=r_rot_control[2],
                upper=r_rot_control[2],
                weight=self.goal_weights[eef].get_expression() + self.get_robot().default_joint_weight,
                expression=c_aa[2])
            self._controllable_constraints = robot.joint_constraints
            self._hard_constraints = robot.hard_constraints
            self.update_observables({self.goal_weights[eef].get_symbol_str(): self.weight})
            self.set_goal({eef: robot.get_eef_position2()[eef]})
        print('make constraints took {}'.format(time() - t))

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        for eef, goal_pos in goal.items():
            self.update_observables(self.goal_eef[eef].get_update_dict(*goal_pos))
            self.update_observables(self.start_eef[eef].get_update_dict(*self.get_robot().get_eef_position2()[eef]))

