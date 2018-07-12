import hashlib
from itertools import chain

import symengine_wrappers as sw
from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.symengine_robot import Robot


class SymEngineController(object):
    def __init__(self, urdf, path_to_functions, default_joint_vel_limit):
        self.urdf = urdf
        self.path_to_functions = path_to_functions
        self._soft_constraints = OrderedDict()
        self.robot = Robot(urdf, default_joint_vel_limit)
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.qp_problem_builder = None

    def is_initialized(self):
        return self.qp_problem_builder is not None

    def set_controlled_joints(self, joint_names):
        """
        :param joint_names:
        :type joint_names: set
        """
        self.controlled_joints.extend(x for x in joint_names if x not in self.controlled_joints)
        self.controlled_joint_symbols = [self.robot.get_joint_symbol(x) for x in self.controlled_joints]
        self.joint_constraints = OrderedDict(((self.robot.get_name(), k), self.robot.joint_constraints[k]) for k in
                                             self.controlled_joints)
        self.hard_constraints = OrderedDict(((self.robot.get_name(), k), self.robot.hard_constraints[k]) for k in
                                            self.controlled_joints if k in self.robot.hard_constraints)

    def init(self, soft_constraints, free_symbols):
        a = ''.join(str(x) for x in sorted(chain(soft_constraints.keys(),
                                                 self.hard_constraints.keys(),
                                                 self.joint_constraints.keys())))
        function_hash = hashlib.md5(a+self.urdf).hexdigest()
        path_to_functions = self.path_to_functions + function_hash
        self.qp_problem_builder = QProblemBuilder(self.joint_constraints,
                                                  self.hard_constraints,
                                                  soft_constraints,
                                                  self.controlled_joint_symbols,
                                                  free_symbols,
                                                  path_to_functions)

    def get_controlled_joints(self):
        return list(self.robot.get_joint_symbols().keys())

    def get_controlled_joint_symbols(self):
        return list(self.robot.get_joint_symbols().values())

    def get_cmd(self, substitutions, nWSR=None):
        next_cmd = self.qp_problem_builder.get_cmd(substitutions, nWSR)
        if next_cmd is None:
            pass
        real_next_cmd = {}
        # TODO use own controlled joints?
        for joint_name in self.get_controlled_joints():
            joint_expr = str(self.robot.get_joint_symbol(joint_name))
            if joint_expr in next_cmd:
                real_next_cmd[joint_name] = next_cmd[joint_expr]
        return real_next_cmd


def joint_position(current_joint, joint_goal, weight, p_gain, max_speed, name):
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
    soft_constraints = OrderedDict()

    err = joint_goal - current_joint
    capped_err = sw.diffable_max_fast(sw.diffable_min_fast(p_gain * err, max_speed), -max_speed)

    soft_constraints[name] = SoftConstraint(lower=capped_err,
                                            upper=capped_err,
                                            weight=weight,
                                            expression=current_joint)
    # add_debug_constraint(soft_constraints, '{} //current_joint//'.format(name), current_joint)
    # add_debug_constraint(soft_constraints, '{} //joint_goal//'.format(name), joint_goal)
    # add_debug_constraint(soft_constraints, '{} //max_speed//'.format(name), max_speed)
    return soft_constraints


def continuous_joint_position(current_joint, change, weight, p_gain, max_speed, name):
    soft_constraints = OrderedDict()

    capped_err = sw.diffable_max_fast(sw.diffable_min_fast(p_gain * change, max_speed), -max_speed)

    soft_constraints[name] = SoftConstraint(lower=capped_err,
                                            upper=capped_err,
                                            weight=weight,
                                            expression=current_joint)
    # add_debug_constraint(soft_constraints, '{} //change//'.format(name), change)
    # add_debug_constraint(soft_constraints, '{} //max_speed//'.format(name), max_speed)
    return soft_constraints


def position_conv(goal_position, current_position, weights=1, trans_gain=3, max_trans_speed=0.3, ns=''):
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
    soft_constraints = OrderedDict()

    trans_error_vector = goal_position - current_position
    trans_error = sw.norm(trans_error_vector)
    trans_scale = sw.diffable_min_fast(trans_error * trans_gain, max_trans_speed)
    trans_control = trans_error_vector / trans_error * trans_scale

    soft_constraints['align {} x position'.format(ns)] = SoftConstraint(lower=trans_control[0],
                                                                        upper=trans_control[0],
                                                                        weight=weights,
                                                                        expression=current_position[0])
    soft_constraints['align {} y position'.format(ns)] = SoftConstraint(lower=trans_control[1],
                                                                        upper=trans_control[1],
                                                                        weight=weights,
                                                                        expression=current_position[1])
    soft_constraints['align {} z position'.format(ns)] = SoftConstraint(lower=trans_control[2],
                                                                        upper=trans_control[2],
                                                                        weight=weights,
                                                                        expression=current_position[2])

    return soft_constraints


def rotation_conv(goal_rotation, current_rotation, current_evaluated_rotation, weights=1,
                  rot_gain=3, max_rot_speed=0.5, ns=''):
    soft_constraints = OrderedDict()
    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * goal_rotation))

    capped_angle = sw.diffable_max_fast(sw.diffable_min_fast(rot_gain * angle, max_rot_speed), -max_rot_speed)

    r_rot_control = axis * capped_angle

    hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints['align {} rotation 0'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                        upper=r_rot_control[0],
                                                                        weight=weights,
                                                                        expression=c_aa[0])
    soft_constraints['align {} rotation 1'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                        upper=r_rot_control[1],
                                                                        weight=weights,
                                                                        expression=c_aa[1])
    soft_constraints['align {} rotation 2'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
                                                                        upper=r_rot_control[2],
                                                                        weight=weights,
                                                                        expression=c_aa[2])
    return soft_constraints


def rotation_conv_slerp(goal_rotation, current_rotation, current_evaluated_rotation, weights=1,
                        rot_gain=3, max_rot_speed=0.5, ns=''):
    soft_constraints = OrderedDict()

    axis, rot_error = sw.axis_angle_from_matrix((current_rotation.T * goal_rotation))

    control = rot_gain * rot_error

    interpolation_value = sw.if_greater_zero(max_rot_speed - control,
                                             1,
                                             max_rot_speed / control)

    intermediate_goal = sw.slerp2(sw.quaternion_from_matrix(current_rotation),
                                  sw.quaternion_from_matrix(goal_rotation),
                                  interpolation_value)

    rm = current_rotation.T * sw.rotation_matrix_from_quaternion(*intermediate_goal)

    axis2, angle2 = sw.axis_angle_from_matrix(rm)

    r_rot_control = current_rotation[:3,:3] * (axis2*angle2)


    hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints['align {} rotation 0'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                        upper=r_rot_control[0],
                                                                        weight=weights,
                                                                        expression=c_aa[0])
    soft_constraints['align {} rotation 1'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                        upper=r_rot_control[1],
                                                                        weight=weights,
                                                                        expression=c_aa[1])
    soft_constraints['align {} rotation 2'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
                                                                        upper=r_rot_control[2],
                                                                        weight=weights,
                                                                        expression=c_aa[2])

    add_debug_constraint(soft_constraints, '{} //debug interpolation_value//'.format(ns), interpolation_value)
    add_debug_constraint(soft_constraints, '{} //debug intermediate_goal[0]//'.format(ns), intermediate_goal[0])
    add_debug_constraint(soft_constraints, '{} //debug intermediate_goal[1]//'.format(ns), intermediate_goal[1])
    add_debug_constraint(soft_constraints, '{} //debug intermediate_goal[2]//'.format(ns), intermediate_goal[2])
    add_debug_constraint(soft_constraints, '{} //debug intermediate_goal[3]//'.format(ns), intermediate_goal[3])
    return soft_constraints

def rotation_conv_slerp2(goal_rotation, current_rotation, current_evaluated_rotation, slerp, weights=1,
                        rot_gain=3, max_rot_speed=0.5, ns=''):
    soft_constraints = OrderedDict()



    hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints['align {} rotation 0'.format(ns)] = SoftConstraint(lower=slerp[0],
                                                                        upper=slerp[0],
                                                                        weight=weights,
                                                                        expression=c_aa[0])
    soft_constraints['align {} rotation 1'.format(ns)] = SoftConstraint(lower=slerp[1],
                                                                        upper=slerp[1],
                                                                        weight=weights,
                                                                        expression=c_aa[1])
    soft_constraints['align {} rotation 2'.format(ns)] = SoftConstraint(lower=slerp[2],
                                                                        upper=slerp[2],
                                                                        weight=weights,
                                                                        expression=c_aa[2])
    return soft_constraints


def link_to_link_avoidance(link_name, current_pose, current_pose_eval, point_on_link, other_point, contact_normal,
                           lower_limit=0.05, upper_limit=1e9, weight=10000):
    soft_constraints = OrderedDict()
    name = '{} to any collision'.format(link_name)

    controllable_point = current_pose * sw.inverse_frame(current_pose_eval) * point_on_link

    dist = (contact_normal.T * (controllable_point - other_point))[0]

    soft_constraints['{} '.format(name)] = SoftConstraint(lower=lower_limit - dist,
                                                          upper=upper_limit,
                                                          weight=weight,
                                                          expression=dist)
    # add_debug_constraint(soft_constraints, '{} //debug dist//'.format(name), dist)
    # add_debug_constraint(soft_constraints, '{} //debug n0//'.format(name), contact_normal[0])
    # add_debug_constraint(soft_constraints, '{} //debug n1//'.format(name), contact_normal[1])
    # add_debug_constraint(soft_constraints, '{} //debug n2//'.format(name), contact_normal[2])
    return soft_constraints


def link_to_link_avoidance_old(link_name, current_pose, current_pose_eval, point_on_link, other_point, lower_limit=0.05,
                               upper_limit=1e9, weight=10000):
    soft_constraints = {}
    name = '{} to any collision'.format(link_name)

    dist = sw.euclidean_distance((current_pose * sw.inverse_frame(current_pose_eval) * point_on_link), other_point)
    soft_constraints['{} cpi'.format(name)] = SoftConstraint(lower=lower_limit - dist,
                                                             upper=upper_limit,
                                                             weight=weight,
                                                             expression=dist)

    return soft_constraints


def add_debug_constraint(d, key, expr):
    d[key] = SoftConstraint(lower=expr,
                            upper=expr,
                            weight=0,
                            expression=1)
