import hashlib
import warnings
from itertools import chain

import symengine_wrappers as sw
from collections import OrderedDict
from giskardpy.qp_problem_builder import QProblemBuilder, SoftConstraint
from giskardpy.symengine_robot import Robot


class SymEngineController(object):
    """
    This class handles constraints and computes joint commands using symengine and qpOases.
    """

    # TODO should anybody who uses this class know about constraints?

    def __init__(self, robot, path_to_functions):
        """
        :type robot: Robot
        :param path_to_functions: location where compiled functions are stored
        :type: str
        """
        self.path_to_functions = path_to_functions
        self.robot = robot
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.soft_constraints = {}
        self.free_symbols = None
        self.qp_problem_builder = None

    def set_controlled_joints(self, joint_names):
        """
        :type joint_names: set
        """
        self.controlled_joints = joint_names
        self.joint_to_symbols_str = OrderedDict((x, self.robot.get_joint_symbol(x)) for x in self.controlled_joints)
        self.joint_constraints = OrderedDict(((self.robot.get_name(), k), self.robot.joint_constraints[k]) for k in
                                             self.controlled_joints)
        self.hard_constraints = OrderedDict(((self.robot.get_name(), k), self.robot.hard_constraints[k]) for k in
                                            self.controlled_joints if k in self.robot.hard_constraints)

    def update_soft_constraints(self, soft_constraints, free_symbols=None):
        """
        Triggers a recompile if the number of soft constraints has changed.
        :type soft_constraints: dict
        :type free_symbols: set
        """
        if free_symbols is not None:
            warnings.warn('use of free_symbols deprecated', DeprecationWarning)
        # TODO bug if soft constraints get replaced, actual amount does not change.
        last_number_of_constraints = len(self.soft_constraints)
        if free_symbols is not None:
            if self.free_symbols is None:
                self.free_symbols = set()
            self.free_symbols.update(free_symbols)
        self.soft_constraints.update(soft_constraints)
        if last_number_of_constraints != len(self.soft_constraints):
            self.qp_problem_builder = None

    def compile(self):
        a = ''.join(str(x) for x in sorted(chain(self.soft_constraints.keys(),
                                                 self.hard_constraints.keys(),
                                                 self.joint_constraints.keys())))
        function_hash = hashlib.md5(a + self.robot.get_hash()).hexdigest()
        path_to_functions = self.path_to_functions + function_hash
        self.qp_problem_builder = QProblemBuilder(self.joint_constraints,
                                                  self.hard_constraints,
                                                  self.soft_constraints,
                                                  self.joint_to_symbols_str.values(),
                                                  self.free_symbols,
                                                  path_to_functions)

    def get_cmd(self, substitutions, nWSR=None):
        """
        Computes joint commands that satisfy constrains given substitutions.
        :param substitutions: maps symbol names as str to floats.
        :type substitutions: dict
        :param nWSR: magic number, if None throws errors, increase this until it stops.
        :type nWSR: int
        :return: maps joint names to command
        :rtype: dict
        """
        try:
            next_cmd = self.qp_problem_builder.get_cmd(substitutions, nWSR)
            if next_cmd is None:
                pass
            return {name: next_cmd[symbol] for name, symbol in self.joint_to_symbols_str.items()}
        except AttributeError:
            self.compile()
            return self.get_cmd(substitutions, nWSR)


def joint_position(current_joint, joint_goal, weight, p_gain, max_speed, name):
    """
    :type current_joint: sw.Symbol
    :type joint_goal: sw.Symbol
    :type weight: sw.Symbol
    :rtype: dict
    """
    soft_constraints = OrderedDict()

    err = joint_goal - current_joint
    # TODO it would be more efficient to safe the max joint vel in hard constraints
    capped_err = sw.diffable_max_fast(sw.diffable_min_fast(p_gain * err, max_speed), -max_speed)

    soft_constraints[name] = SoftConstraint(lower=capped_err,
                                            upper=capped_err,
                                            weight=weight,
                                            expression=current_joint)
    # add_debug_constraint(soft_constraints, '{} //current_joint//'.format(name), current_joint)
    # add_debug_constraint(soft_constraints, '{} //joint_goal//'.format(name), joint_goal)
    # add_debug_constraint(soft_constraints, '{} //max_speed//'.format(name), max_speed)
    return soft_constraints


def continuous_joint_position(current_joint, rotation_distance, weight, p_gain, max_speed, constraint_name):
    """
    :type current_joint: sw.Symbol
    :type rotation_distance: sw.Symbol
    :type weight: sw.Symbol
    :type p_gain: sw.Symbol
    :param max_speed: in rad/s or m/s depending on joint type.
    :type max_speed: sw.Symbol
    :type constraint_name: str
    :dict:
    """
    # TODO almost the same as joint_position
    soft_constraints = OrderedDict()

    capped_err = sw.diffable_max_fast(sw.diffable_min_fast(p_gain * rotation_distance, max_speed), -max_speed)

    soft_constraints[constraint_name] = SoftConstraint(lower=capped_err,
                                                       upper=capped_err,
                                                       weight=weight,
                                                       expression=current_joint)
    # add_debug_constraint(soft_constraints, '{} //change//'.format(name), change)
    # add_debug_constraint(soft_constraints, '{} //max_speed//'.format(name), max_speed)
    return soft_constraints


def position_conv(goal_position, current_position, weights=1, trans_gain=3, max_trans_speed=0.3, ns=''):
    """
    Creates soft constrains which computes how current_position has to change to become goal_position.
    :param goal_position: 4x1 symengine Matrix.
    :type goal_position: sw.Matrix
    :param current_position: 4x1 symengine Matrix. Describes fk with joint positions.
    :type current_position: sw.Matrix
    :param weights: how important are these constraints
    :type weights: sw.Symbol
    :param trans_gain: how was max_trans_speed is reached.
    :type trans_gain: sw.Symbol
    :param max_trans_speed: maximum speed in m/s
    :type max_trans_speed: sw.Symbol
    :param ns: some string to make constraint names unique
    :type ns: str
    :return: contains the constraints
    :rtype: dict
    """
    soft_constraints = OrderedDict()

    trans_error_vector = goal_position - current_position
    trans_error = sw.norm(trans_error_vector)
    trans_scale = sw.diffable_min_fast(trans_error * trans_gain, max_trans_speed)
    trans_control = trans_error_vector / trans_error * trans_scale

    soft_constraints[u'align {} x position'.format(ns)] = SoftConstraint(lower=trans_control[0],
                                                                         upper=trans_control[0],
                                                                         weight=weights,
                                                                         expression=current_position[0])
    soft_constraints[u'align {} y position'.format(ns)] = SoftConstraint(lower=trans_control[1],
                                                                         upper=trans_control[1],
                                                                         weight=weights,
                                                                         expression=current_position[1])
    soft_constraints[u'align {} z position'.format(ns)] = SoftConstraint(lower=trans_control[2],
                                                                         upper=trans_control[2],
                                                                         weight=weights,
                                                                         expression=current_position[2])

    return soft_constraints


def rotation_conv(goal_rotation, current_rotation, current_evaluated_rotation, weights=1,
                  rot_gain=3, max_rot_speed=0.5, ns=''):
    """
    Creates soft constrains which computes how current_rotation has to change to become goal_rotation.
    :param goal_rotation: 4x4 symengine Matrix.
    :type goal_rotation: sw.Matrix
    :param current_rotation: 4x4 symengine Matrix. Describes current rotation with joint positions
    :type current_rotation: sw.Matrix
    :param current_evaluated_rotation: 4x4 symengine Matrix. contains the evaluated current rotation.
    :type current_evaluated_rotation: sw.Matrix
    :param weights: how important these constraints are
    :type weights: sw.Symbol
    :param rot_gain: how quickly max_rot_speed is reached.
    :type rot_gain: sw.Symbol
    :param max_rot_speed: maximum rotation speed in rad/s
    :type max_rot_speed: sw.Symbol
    :param ns: some string to make the constraint names unique
    :return: contains the constraints.
    :rtype: dict
    """
    soft_constraints = OrderedDict()
    axis, angle = sw.axis_angle_from_matrix_stable((current_rotation.T * goal_rotation))

    capped_angle = sw.diffable_max_fast(sw.diffable_min_fast(rot_gain * angle, max_rot_speed), -max_rot_speed)

    r_rot_control = axis * capped_angle

    hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints[u'align {} rotation 0'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                         upper=r_rot_control[0],
                                                                         weight=weights,
                                                                         expression=c_aa[0])
    soft_constraints[u'align {} rotation 1'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                         upper=r_rot_control[1],
                                                                         weight=weights,
                                                                         expression=c_aa[1])
    soft_constraints[u'align {} rotation 2'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
                                                                         upper=r_rot_control[2],
                                                                         weight=weights,
                                                                         expression=c_aa[2])
    return soft_constraints


def rotation_conv_slerp(goal_rotation, current_rotation, current_evaluated_rotation, weights=1,
                        rot_gain=3, max_rot_speed=0.5, ns=''):
    """
    Creates soft constrains which computes how current_rotation has to change to become goal_rotation.
    :param goal_rotation: 4x4 symengine Matrix.
    :type goal_rotation: sw.Matrix
    :param current_rotation: 4x4 symengine Matrix. Describes current rotation with joint positions
    :type current_rotation: sw.Matrix
    :param current_evaluated_rotation: 4x4 symengine Matrix. contains the evaluated current rotation.
    :type current_evaluated_rotation: sw.Matrix
    :param weights: how important these constraints are
    :type weights: sw.Symbol
    :param rot_gain: how quickly max_rot_speed is reached.
    :type rot_gain: sw.Symbol
    :param max_rot_speed: maximum rotation speed in rad/s
    :type max_rot_speed: sw.Symbol
    :param ns: some string to make the constraint names unique
    :return: contains the constraints.
    :rtype: dict
    """
    soft_constraints = OrderedDict()

    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * goal_rotation))
    angle = sw.diffable_abs(angle)

    capped_angle = sw.diffable_min_fast(max_rot_speed / (rot_gain * angle), 1)

    q1 = sw.quaternion_from_matrix(current_rotation)
    q2 = sw.quaternion_from_matrix(goal_rotation)
    intermediate_goal = sw.diffable_slerp(q1, q2, capped_angle)
    axis, angle = sw.axis_angle_from_quaternion(*sw.quaternion_diff(q1, intermediate_goal))
    # intermediate_goal = sw.rotation_matrix_from_quaternion(*intermediate_goal)
    # axis, angle = sw.axis_angle_from_matrix((current_rotation.T * intermediate_goal))
    r_rot_control = axis * angle

    # axis, angle = sw.axis_angle_from_matrix((current_rotation.T * goal_rotation))

    hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
    axis, angle = sw.axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints[u'align {} rotation 0'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                         upper=r_rot_control[0],
                                                                         weight=weights,
                                                                         expression=c_aa[0])
    soft_constraints[u'align {} rotation 1'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                         upper=r_rot_control[1],
                                                                         weight=weights,
                                                                         expression=c_aa[1])
    soft_constraints[u'align {} rotation 2'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
                                                                         upper=r_rot_control[2],
                                                                         weight=weights,
                                                                         expression=c_aa[2])
    return soft_constraints



def link_to_link_avoidance(link_name, current_pose, current_pose_eval, point_on_link, other_point, contact_normal,
                           lower_limit=0.05, upper_limit=1e9, weight=10000):
    """
    Pushes a robot link away from another point.
    :type link_name: str
    :param current_pose: 4x4 symengine matrix describing the fk to the link with joint positions.
    :type current_pose: sw.Matrix
    :param current_pose_eval: 4x4 symengine matrix which contains the pose of the link. The entries should only be one symbol
                                which get directly replaced with the fk.
    :type current_pose_eval: sw.Matrix
    :param point_on_link: 4x1 symengine Matrix. Point on the link in root frame.
    :type point_on_link: sw.Matrix
    :param other_point: 4x1 symengine Matrix. Position of the other point in root frame.
    :type other_point: sw.Matrix
    :param contact_normal: 4x1 symengine Matrix. Vector pointing from the other point to the contact point on the link.
    :type contact_normal: sw.Matrix
    :param lower_limit: minimal allowed distance to the other point.
    :type lower_limit: sw.Symbol
    :param upper_limit: maximum distance allowed to the other point.
    :type upper_limit: sw.Symbol
    :param weight: How important this constraint is.
    :type weight: sw.Symbol
    :return: contains the soft constraint.
    :rtype: dict
    """
    soft_constraints = OrderedDict()
    name = u'{} to any collision'.format(link_name)

    controllable_point = current_pose * sw.inverse_frame(current_pose_eval) * point_on_link

    dist = (contact_normal.T * (controllable_point - other_point))[0]

    soft_constraints[u'{} '.format(name)] = SoftConstraint(lower=lower_limit - dist,
                                                           upper=upper_limit,
                                                           weight=weight,
                                                           expression=dist)
    # add_debug_constraint(soft_constraints, '{} //debug dist//'.format(name), dist)
    # add_debug_constraint(soft_constraints, '{} //debug n0//'.format(name), contact_normal[0])
    # add_debug_constraint(soft_constraints, '{} //debug n1//'.format(name), contact_normal[1])
    # add_debug_constraint(soft_constraints, '{} //debug n2//'.format(name), contact_normal[2])
    return soft_constraints


def add_debug_constraint(d, key, expr):
    """
    If you want to see an arbitrary evaluated expression in the matrix use this.
    These softconstraints will not influence anything.
    :param d: a dict where the softcontraint will be added to
    :type: dict
    :param key: a name to identify the debug soft contraint
    :type key: str
    :type expr: sw.Symbol
    """
    d[key] = SoftConstraint(lower=expr,
                            upper=expr,
                            weight=0,
                            expression=1)
