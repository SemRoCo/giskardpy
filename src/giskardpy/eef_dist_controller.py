from time import time

from giskardpy import USE_SYMENGINE
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput, FrameInput

if USE_SYMENGINE:
    import giskardpy.symengine_wrappers as spw
else:
    import giskardpy.sympy_wrappers as spw


class EEFDiffController(QPController):
    def __init__(self, robot, builder_backend=None, weight=1):
        self.weight = weight
        super(EEFDiffController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_diff = FrameInput(prefix='', suffix='goal')
        self.goal_weights = ScalarInput(prefix='', suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        t = time()
        eef1 = robot.end_effectors[0]
        eef1_frame = robot.frames[eef1]
        eef2 = robot.end_effectors[1]
        eef2_frame = robot.frames[eef2]

        eef_diff_pos = spw.pos_of(eef1_frame) - spw.pos_of(eef2_frame)
        goal_pos = self.goal_diff.get_position()
        dist = spw.norm((eef_diff_pos) - goal_pos)

        eef_diff_rot = spw.rot_of(eef1_frame)[:3, :3].T * spw.rot_of(eef2_frame)[:3, :3]

        goal_rot = self.goal_diff.get_rotation()

        goal_r = goal_rot[:3, :3].reshape(9, 1)
        dist_r = spw.norm(eef_diff_rot.reshape(9, 1) - goal_r)

        self._soft_constraints['align eefs position'] = SoftConstraint(lower=-dist,
                                                                       upper=-dist,
                                                                       weight=self.goal_weights.get_expression(),
                                                                       expression=dist)
        self._soft_constraints['align eefs rotation'] = SoftConstraint(lower=-dist_r,
                                                                       upper=-dist_r,
                                                                       weight=self.goal_weights.get_expression(),
                                                                       expression=dist_r)
        self._controllable_constraints = robot.joint_constraints
        self._hard_constraints = robot.hard_constraints
        self.update_observables({self.goal_weights.get_symbol_str(): self.weight})
        print('make constraints took {}'.format(time() - t))

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        self.update_observables(self.goal_diff.get_update_dict(*goal))
