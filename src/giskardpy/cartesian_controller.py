from time import time

from tf.transformations import quaternion_about_axis, quaternion_matrix
from numpy import pi
from giskardpy import USE_SYMENGINE
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput, FrameInput

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw


class CartesianController(QPController):
    def __init__(self, robot, builder_backend=None, weight=1, gain=30, threshold_value=.05):
        self.weight = weight
        self.default_gain = gain
        self.default_threshold = threshold_value

        self.max_trans_speed = 0.3  # m/s
        # self.default_trans_threshold = 0
        self.default_trans_gain = 3

        # self.default_rot_threshold = 0.2
        self.max_rot_speed = 0.5  # rad/s
        self.default_rot_gain = 3

        super(CartesianController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.start_eef = {}
        self.goal_weights = {}
        self.gain = ScalarInput(prefix='gain')
        self.threshold_value = ScalarInput(prefix='threshold_value')
        for eef in robot.end_effectors:
            self.goal_eef[eef] = FrameInput(prefix=eef, suffix='goal')
            self.start_eef[eef] = FrameInput(prefix=eef, suffix='start')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        t = time()
        for eef in robot.end_effectors:
            # --------------------------------------------
            # class testmuh(object):
            #     def __init__(self, robot):
            #         self.robot = robot
            #
            #     def __call__(self, observables):
            #         return 5
            # self.get_state()['testmuh'] = testmuh(self.get_robot())
            # --------------------------------------------

            start_position = self.start_eef[eef].get_position()
            start_rotation = self.start_eef[eef].get_rotation()

            current_pose = robot.frames[eef]
            current_position = spw.pos_of(current_pose)
            current_rotation = spw.rot_of(current_pose)
            current_quaternion = robot.q_rot[eef]

            goal_position = self.goal_eef[eef].get_position()
            goal_rotation = self.goal_eef[eef].get_rotation()
            goal_quaternion = self.goal_eef[eef].get_quaternion()
            # goal_r = goal_rotation[:3,:3].reshape(9,1)

            # pos control ----------------------------------------------------------------------------------------------
            # goal_trans = current_position - goal_position
            trans_error_vector = goal_position - current_position
            trans_error = spw.norm(trans_error_vector)
            # trans_scale = spw.Min(1, self.threshold_value.get_expression() / trans_error)
            # trans_control = trans_error_vector  * (self.gain.get_expression() * trans_scale)
            #
            trans_scale = spw.fake_Min(trans_error * self.default_trans_gain, self.max_trans_speed)
            trans_control = trans_error_vector / trans_error * trans_scale

            # dist = spw.norm(spw.pos_of(current_pose) - goal_position)
            # dist = spw.Min(self.threshold_value.get_expression(), dist)
            # dist *= self.gain.get_expression()

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

            # rot control ----------------------------------------------------------------------------------------------

            # quaternion + cosine distance
            # dist_r = spw.sqrt(spw.fake_Min(spw.cosine_distance(current_quaternion, goal_quaternion),
            #                                spw.cosine_distance(current_quaternion, -goal_quaternion)))
            # dist_r_control = spw.fake_Min(dist_r * self.default_rot_gain, self.max_rot_speed)
            #
            # self._soft_constraints['align {} rotation'.format(eef)] = SoftConstraint(lower=-dist_r_control,
            #                                                                          upper=-dist_r_control,
            #                                                                          weight=self.goal_weights[
            #                                                                              eef].get_expression(),
            #                                                                          expression=dist_r)

            # georg style

            # dist_r = spw.rotation_distance(current_rotation, goal_rotation)
            # dist_r_control = spw.Min(dist_r * self.default_rot_gain, self.max_rot_speed)

            axis, angle = spw.axis_angle_from_matrix((current_rotation.T * goal_rotation))
            capped_angle = spw.fake_Min(angle * self.default_rot_gain, self.max_rot_speed)
            axis = current_rotation * spw.vec3(*axis)
            r_rot_control = axis * capped_angle

            hack = spw.rotation3_axis_angle([0,0,1],0.0001)

            axis, angle = spw.axis_angle_from_matrix((current_rotation.T * (start_rotation*hack)).T)
            # axis, angle = spw.axis_angle_from_matrix((current_rotation.T * current_rotation).T)
            c_aa = current_rotation[:3,:3]*(axis*angle)

            # c_axis, c_angle = spw.axis_angle_from_matrix(spw.eye(4))
            # c_aa = c_axis * c_angle
            # c_aa = current_rotation * spw.vec3(0,0,0.1)



            # g_axis, g_angle = spw.axis_angle_from_matrix(goal_rotation)
            # g_aa = g_axis * g_angle
            #
            # r_rot_control = g_aa - c_aa

            # diff_angle = spw.norm(r_rot_control)
            # capped_angle = spw.fake_Min(diff_angle * self.default_rot_gain, self.max_rot_speed)
            # r_rot_control = r_rot_control / diff_angle * capped_angle

            self._soft_constraints['align {} rotation 0'.format(eef)] = SoftConstraint(
                lower=r_rot_control[0],
                upper=r_rot_control[0],
                weight=self.goal_weights[eef].get_expression()+self.get_robot().default_joint_weight,
                expression=c_aa[0])
            self._soft_constraints['align {} rotation 1'.format(eef)] = SoftConstraint(
                lower=r_rot_control[1],
                upper=r_rot_control[1],
                weight=self.goal_weights[eef].get_expression()+self.get_robot().default_joint_weight,
                expression=c_aa[1])
            self._soft_constraints['align {} rotation 2'.format(eef)] = SoftConstraint(
                lower=r_rot_control[2],
                upper=r_rot_control[2],
                weight=self.goal_weights[eef].get_expression()+self.get_robot().default_joint_weight,
                expression=c_aa[2])


            self._controllable_constraints = robot.joint_constraints
            self._hard_constraints = robot.hard_constraints
            self.update_observables({self.goal_weights[eef].get_symbol_str(): self.weight})
            self.set_goal({eef: robot.get_eef_position2()[eef]})
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
            self.update_observables(self.start_eef[eef].get_update_dict(*self.get_robot().get_eef_position2()[eef]))
