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
        super(CartesianLineController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.goal_weights = {}
        for eef in robot.end_effectors:
            self.goal_eef[eef] = FrameInput(prefix=eef, suffix='goal')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        t = time()
        for eef in robot.end_effectors:
            start_pose = robot.get_eef_position()[eef]
            start_position = spw.pos_of(start_pose)

            current_pose = robot.frames[eef]
            current_position = spw.pos_of(current_pose)
            current_rotation = spw.rot_of(current_pose)[:3, :3]

            goal_position = self.goal_eef[eef].get_position()
            goal_rotation = self.goal_eef[eef].get_rotation()[:3, :3]

            # pos control
            dist = spw.norm(spw.pos_of(current_pose) - goal_position)

            # line
            x0 = current_position[:-1, :]
            x1 = spw.Matrix((start_position[:-1, :] + np.random.random((3,1))*0.005).astype(float).tolist())
            x2 = goal_position[:-1, :]
            dist_to_line = spw.norm(spw.cross((x0 - x1), (x0 - x2))) / spw.norm(x2 - x1)
            dist_to_line = dist_to_line ** 2

            # rot control
            dist_r = spw.norm(current_rotation.reshape(9, 1) - goal_rotation.reshape(9, 1))

            self._soft_constraints['align {} position'.format(eef)] = SoftConstraint(lower=-dist,
                                                                                     upper=-dist,
                                                                                     weight=self.goal_weights[
                                                                                         eef].get_expression(),
                                                                                     expression=dist)
            self._soft_constraints['align {} rotation'.format(eef)] = SoftConstraint(lower=-dist_r,
                                                                                     upper=-dist_r,
                                                                                     weight=self.goal_weights[
                                                                                         eef].get_expression(),
                                                                                     expression=dist_r)
            self._soft_constraints['{} stay on line'.format(eef)] = SoftConstraint(lower=-dist_to_line,
                                                                                   upper=-dist_to_line,
                                                                                   weight=self.goal_weights[
                                                                                              eef].get_expression() * 100,
                                                                                   expression=dist_to_line)
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
