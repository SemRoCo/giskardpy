from giskardpy.qpcontroller import QPController
from giskardpy.sympy_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import Point3Input, ControllerInputArray, ScalarInput, FrameInput


class CartesianController(QPController):
    def __init__(self, robot, weight=1):
        self.weight = weight
        super(CartesianController, self).__init__(robot)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = {}
        self.goal_weights = {}
        for eef in robot.end_effectors:
            self.goal_eef[eef] = FrameInput(prefix=eef, suffix='goal')
            self.goal_weights[eef] = ScalarInput(prefix=eef, suffix='sc_w')

    # @profile
    def make_constraints(self, robot):
        for eef in robot.end_effectors:
            eef_frame = robot.frames[eef]
            goal_pos = self.goal_eef[eef].get_position()
            dist = norm(pos_of(eef_frame) - goal_pos)

            goal_rot = self.goal_eef[eef].get_rotation()

            eef_r = rot_of(eef_frame)[:3,:3].reshape(9,1)
            goal_r = goal_rot[:3,:3].reshape(9,1)
            dist_r = norm(eef_r-goal_r)
            #
            # eef_r = rot_of(eef_frame)[:3,:3]#.reshape(9,1)
            # goal_r = rot_of(goal_rot)[:3,:3].T#.reshape(9,1)
            # dist_r = sp.acos((sp.trace(goal_r.T*eef_r)-1)/2)

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
            self._controllable_constraints = robot.joint_constraints
            self._hard_constraints = robot.hard_constraints
            self.update_observables({self.goal_weights[eef].get_symbol_str(): self.weight})
            self.set_goal({eef: robot.get_eef_position2()[eef]})

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        for eef, goal_pos in goal.items():
            self.update_observables(self.goal_eef[eef].get_update_dict(*goal_pos))
