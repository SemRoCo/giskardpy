from time import time

from giskardpy import USE_SYMENGINE
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput, FrameInput

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw


class CartesianController(QPController):
    def __init__(self, robot, builder_backend=None, weight=1, gain=3, threshold_value=.05):
        self.weight = weight
        self.default_gain = gain
        self.default_threshold = threshold_value
        super(CartesianController, self).__init__(robot, builder_backend)

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
            # goal_trans = current_position - goal_position
            trans_error_vector = goal_position - current_position
            trans_error = spw.norm(trans_error_vector)
            trans_scale = spw.Min(1, self.threshold_value.get_expression() / trans_error)
            trans_scaled_error = trans_scale * trans_error
            trans_control = trans_error_vector * (self.gain.get_expression() * trans_scale)

            # dist = spw.norm(spw.pos_of(current_pose) - goal_position)
            # dist = spw.Min(self.threshold_value.get_expression(), dist)
            # dist *= self.gain.get_expression()

            # rot control ----------------------------------------------------------------------------------------------
            dist_r = spw.norm(current_rotation.reshape(9, 1) - goal_rotation.reshape(9, 1))


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
            # self._soft_constraints['align {} rotation'.format(eef)] = SoftConstraint(lower=-dist_r,
            #                                                                          upper=-dist_r,
            #                                                                          weight=self.goal_weights[
            #                                                                              eef].get_expression(),
            #                                                                          expression=dist_r)
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
