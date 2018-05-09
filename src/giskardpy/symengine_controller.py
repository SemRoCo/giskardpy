from symengine import Symbol

import symengine_wrappers as sw
from collections import OrderedDict

from giskardpy.input_system import JointStatesInput
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.symengine_robot import Robot


class SymEngineController(object):
    def __init__(self, urdf):
        self._soft_constraints = OrderedDict()
        self.robot = Robot(urdf)
        self.controlled_joints = set()
        self.hard_constraints = {}
        self.joint_constraints = {}

    def set_controlled_joints(self, joint_names):
        """
        :param joint_names:
        :type joint_names: set
        """
        # TODO might have to get it from a topic or something
        self.controlled_joints.update(joint_names)
        self.controlled_joint_symbols = [self.robot.get_joint_symbol_map().joint_map[x] for x in
                                         self.controlled_joints]
        self.joint_constraints = {k: self.robot.joint_constraints[k] for k in self.controlled_joints}
        self.hard_constraints = {k: self.robot.hard_constraints[k] for k in self.controlled_joints if
                                 k in self.robot.hard_constraints}

    def init(self, soft_constraints, free_symbols):
        self.qp_problem_builder = QProblemBuilder(self.joint_constraints,
                                                  self.hard_constraints,
                                                  soft_constraints,
                                                  self.controlled_joint_symbols,
                                                  free_symbols)

    def get_controlled_joints(self):
        return list(self.robot.joint_states_input.joint_map.keys())

    def get_controlled_joint_symbols(self):
        return list(self.robot.joint_states_input.joint_map.values())

    def get_cmd(self, substitutions):
        next_cmd = self.qp_problem_builder.get_cmd(substitutions)
        if next_cmd is None:
            pass
        real_next_cmd = {}
        for joint_name in self.get_controlled_joints():
            joint_expr = str(self.robot.joint_states_input.joint_map[joint_name])
            if joint_expr in next_cmd:
                real_next_cmd[joint_name] = next_cmd[joint_expr]
        return real_next_cmd



def joint_position(current_joint, joint_goal, weight):
    """
    :param current_joint:
    :type current_joint: Symbol
    :param joint_goal:
    :type joint_goal: Symbol
    :param weight:
    :type weight: Symbol
    :return:
    :rtype: dict
    """
    return SoftConstraint(lower=joint_goal - current_joint,
                          upper=joint_goal - current_joint,
                          weight=weight,
                          expression=current_joint)


def position_conv(goal_position, current_position, weights=(1, 1, 1), trans_gain=3, max_trans_speed=0.3, ns=''):
    """
    :param goal_position:
    :type goal_position: giskardpy.input_system.FrameInput
    :param current_position:
    :type current_position: giskardpy.input_system.FrameInput
    :param weights:
    :type weights:
    :param trans_gain:
    :param max_trans_speed:
    :param ns:
    :return:
    """
    soft_constraints = {}

    trans_error_vector = goal_position - current_position
    trans_error = sw.norm(trans_error_vector)
    trans_scale = sw.fake_Min(trans_error * trans_gain, max_trans_speed)
    trans_control = trans_error_vector / trans_error * trans_scale

    soft_constraints['align {} x position'.format(ns)] = SoftConstraint(lower=trans_control[0],
                                                                        upper=trans_control[0],
                                                                        weight=weights[0],
                                                                        expression=current_position[0])
    soft_constraints['align {} y position'.format(ns)] = SoftConstraint(lower=trans_control[1],
                                                                        upper=trans_control[1],
                                                                        weight=weights[1],
                                                                        expression=current_position[1])
    soft_constraints['align {} z position'.format(ns)] = SoftConstraint(lower=trans_control[2],
                                                                        upper=trans_control[2],
                                                                        weight=weights[2],
                                                                        expression=current_position[2])

    return soft_constraints


def rotation_conv(goal_rotation, current_rotation, current_evaluated_rotation, weights=(1, 1, 1),
                  rot_gain=3, max_rot_speed=0.5, ns=''):
    soft_constraints = {}
    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * goal_rotation))
    capped_angle = sw.fake_Min(angle * rot_gain, max_rot_speed)
    r_rot_control = axis * capped_angle

    hack = sw.rotation3_axis_angle([0, 0, 1], 0.0001)

    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints['align {} rotation 0'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                        upper=r_rot_control[0],
                                                                        weight=weights[0],
                                                                        expression=c_aa[0])
    soft_constraints['align {} rotation 1'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                        upper=r_rot_control[1],
                                                                        weight=weights[1],
                                                                        expression=c_aa[1])
    soft_constraints['align {} rotation 2'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
                                                                        upper=r_rot_control[2],
                                                                        weight=weights[2],
                                                                        expression=c_aa[2])
    return soft_constraints


def link_to_any_avoidance(link_name, current_pose, current_pose_eval, point_on_link, other_point, lower_limit=0.05,
                          upper_limit=1e9, weight=10000):
    soft_constraints = {}
    name = '{} to any collision'.format(link_name)

    dist = sw.euclidean_distance((current_pose * sw.inverse_frame(current_pose_eval) * point_on_link), other_point)
    soft_constraints['{} x'.format(name)] = SoftConstraint(lower=lower_limit - dist,
                                                           upper=upper_limit,
                                                           weight=weight,
                                                           expression=dist)

    return soft_constraints
