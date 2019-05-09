from collections import OrderedDict

from geometry_msgs.msg import PoseStamped
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message

import symengine_wrappers as sw
from giskardpy.identifier import robot_identifier, world_identifier, js_identifier, constraints_identifier, \
    fk_identifier
from giskardpy.input_system import FrameInput
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.tfwrapper import transform_pose


class Constraint(object):
    def __init__(self, god_map, prefix):
        self.god_map = god_map
        self.prefix = prefix

    def safe_params_on_god_map(self, **kwargs):
        constraints = self.get_god_map().safe_get_data(constraints_identifier)
        constraints[str(self)] = kwargs
        self.get_god_map().safe_set_data(constraints_identifier, constraints)

    def get_constraint(self, **kwargs):
        pass

    def get_identifier(self):
        return self.prefix + [str(self)]

    def get_symbol(self, name):
        key = self.get_identifier() + [name]
        return self.god_map.to_symbol(key)

    def get_world_object_pose(self, object_name, link_name):
        pass

    def get_distance(self, body_a, link_a, body_b, link_b):
        pass

    def get_contact_normal_on_b(self, body_a, link_a, body_b, link_b):
        pass

    def get_closest_point_on_a(self, body_a, link_a, body_b, link_b):
        pass

    def get_closest_point_on_b(self, body_a, link_a, body_b, link_b):
        pass

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def get_world(self):
        """
        :rtype: giskardpy.world.World
        """
        return self.get_god_map().safe_get_data(world_identifier)

    def get_robot(self):
        """
        :rtype: giskardpy.symengine_robot.Robot
        """
        return self.get_god_map().safe_get_data(robot_identifier)

    def joint_position_expr(self, joint_name):
        key = js_identifier + [joint_name, u'position']
        return self.god_map.to_symbol(key)

    def __str__(self):
        return self.__class__.__name__


class JointPosition(Constraint):

    def safe_params_on_god_map(self, **kwargs):
        self.joint_name = kwargs[u'joint_name']
        super(JointPosition, self).safe_params_on_god_map(**kwargs)

    def get_constraint(self, joint_name, **kwargs):
        """
        example:
        name='JointPosition'
        parameter_value_pair='{
            "joint_name": "torso_lift_joint", #required
            "goal_position": 0, #required
            "weight": 1, #optional
            "gain": 10, #optional -- error is multiplied with this value
            "max_speed": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :param joint_name:
        :param kwargs:
        :return:
        """

        current_joint = self.joint_position_expr(joint_name)

        joint_goal = self.get_symbol(u'goal_position')
        weight = self.get_symbol(u'weight')
        p_gain = self.get_symbol(u'gain')
        max_speed = self.get_symbol(u'max_speed')

        soft_constraints = OrderedDict()

        if self.get_robot().is_joint_continuous(joint_name):
            err = sw.shortest_angular_distance(current_joint, joint_goal)
        else:
            err = joint_goal - current_joint
        capped_err = sw.diffable_max_fast(sw.diffable_min_fast(p_gain * err, max_speed), -max_speed)

        soft_constraints[str(self)] = SoftConstraint(lower=capped_err,
                                                     upper=capped_err,
                                                     weight=weight,
                                                     expression=current_joint)
        return soft_constraints

    def __str__(self):
        return u'{}/{}'.format(self.__class__.__name__, self.joint_name)


class CartesianConstraint(Constraint):
    def __str__(self):
        return u'{}/{}/{}'.format(self.__class__.__name__, self.root, self.tip)

    def safe_params_on_god_map(self, **kwargs):
        self.root = kwargs[u'root']
        self.tip = kwargs[u'tip']
        goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped', kwargs[u'goal'])
        goal = transform_pose(self.root, goal)
        kwargs[u'goal'] = goal
        super(CartesianConstraint, self).safe_params_on_god_map(**kwargs)

    def get_goal_pose(self, name):
        return FrameInput(self.get_god_map().to_symbol,
                          translation_prefix=constraints_identifier +
                                             [str(self),
                                              name,
                                              u'pose',
                                              u'position'],
                          rotation_prefix=constraints_identifier +
                                          [str(self),
                                           name,
                                           u'pose',
                                           u'orientation']).get_frame()

    def get_fk(self, root, tip):
        return self.get_robot().get_fk_expression(root, tip)

    def get_fk_evaluated(self, root, tip):
        return FrameInput(self.get_god_map().to_symbol,
                          translation_prefix=fk_identifier +
                                             [(root, tip),
                                              u'pose',
                                              u'position'],
                          rotation_prefix=fk_identifier +
                                          [(root, tip),
                                           u'pose',
                                           u'orientation']).get_frame()


class CartesianPosition(CartesianConstraint):

    def get_constraint(self, root, tip, **kwargs):
        """
        example:
        name='CartesianPosition'
        parameter_value_pair='{
            "root": "base_footprint", #required
            "tip": "r_gripper_tool_frame", #required
            "goal_position": {"header":
                                {"stamp":
                                    {"secs": 0,
                                    "nsecs": 0},
                                "frame_id": "",
                                "seq": 0},
                            "pose": {"position":
                                        {"y": 0.0,
                                        "x": 0.0,
                                        "z": 0.0},
                                    "orientation": {"y": 0.0,
                                                    "x": 0.0,
                                                    "z": 0.0,
                                                    "w": 0.0}
                                    }
                            }', #required
            "weight": 1, #optional
            "gain": 3, #optional -- error is multiplied with this value
            "max_speed": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :param root:
        :param tip:
        :param goal_position:
        :param weights:
        :param gain:
        :param max_trans_speed:
        :param ns:
        :return:
        """

        goal_position = sw.position_of(self.get_goal_pose(u'goal'))
        weight = self.get_symbol(u'weight')
        gain = self.get_symbol(u'gain')
        max_speed = self.get_symbol(u'max_speed')

        current_position = sw.position_of(self.get_fk(root, tip))

        soft_constraints = OrderedDict()

        trans_error_vector = goal_position - current_position
        trans_error = sw.norm(trans_error_vector)
        trans_scale = sw.diffable_min_fast(trans_error * gain, max_speed)
        trans_control = trans_error_vector / trans_error * trans_scale

        soft_constraints[u'position x: {}/{}'.format(root, tip)] = SoftConstraint(lower=trans_control[0],
                                                                                  upper=trans_control[0],
                                                                                  weight=weight,
                                                                                  expression=current_position[0])
        soft_constraints[u'position y: {}/{}'.format(root, tip)] = SoftConstraint(lower=trans_control[1],
                                                                                  upper=trans_control[1],
                                                                                  weight=weight,
                                                                                  expression=current_position[1])
        soft_constraints[u'position z: {}/{}'.format(root, tip)] = SoftConstraint(lower=trans_control[2],
                                                                                  upper=trans_control[2],
                                                                                  weight=weight,
                                                                                  expression=current_position[2])
        add_debug_constraint(soft_constraints, 'pos x', current_position[0])
        add_debug_constraint(soft_constraints, 'pos y', current_position[1])
        add_debug_constraint(soft_constraints, 'pos z', current_position[2])
        add_debug_constraint(soft_constraints, 'pos z trans_error_vector', trans_error_vector[2])
        add_debug_constraint(soft_constraints, 'pos  trans_error', trans_error)
        add_debug_constraint(soft_constraints, 'pos trans_scale', trans_scale)
        add_debug_constraint(soft_constraints, 'pos gain', gain)
        add_debug_constraint(soft_constraints, 'pos max_speed', max_speed)
        add_debug_constraint(soft_constraints, 'pos z goal_position', goal_position[2])

        return soft_constraints


class CartesianOrientation(CartesianConstraint):

    def get_constraint(self, root, tip, **kwargs):
        """
        example:
        name='CartesianPosition'
        parameter_value_pair='{
            "root": "base_footprint", #required
            "tip": "r_gripper_tool_frame", #required
            "goal_position": {"header":
                                {"stamp":
                                    {"secs": 0,
                                    "nsecs": 0},
                                "frame_id": "",
                                "seq": 0},
                            "pose": {"position":
                                        {"y": 0.0,
                                        "x": 0.0,
                                        "z": 0.0},
                                    "orientation": {"y": 0.0,
                                                    "x": 0.0,
                                                    "z": 0.0,
                                                    "w": 0.0}
                                    }
                            }', #required
            "weight": 1, #optional
            "gain": 3, #optional -- error is multiplied with this value
            "max_speed": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :param root:
        :param tip:
        :param goal_position:
        :param weights:
        :param gain:
        :param max_speed:
        :param ns:
        :return:
        """
        goal_rotation = sw.rotation_of(self.get_goal_pose(u'goal'))
        weight = self.get_symbol(u'weight')
        gain = self.get_symbol(u'gain')
        max_speed = self.get_symbol(u'max_speed')

        current_rotation = sw.rotation_of(self.get_fk(root, tip))
        current_evaluated_rotation = sw.rotation_of(self.get_fk_evaluated(root, tip))

        soft_constraints = OrderedDict()
        axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * goal_rotation))

        capped_angle = sw.diffable_max_fast(sw.diffable_min_fast(gain * angle, max_speed), -max_speed)

        r_rot_control = axis * capped_angle

        hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

        axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
        c_aa = (axis * angle)

        soft_constraints[u'rotation 0: {}/{}'.format(root, tip)] = SoftConstraint(lower=r_rot_control[0],
                                                                                  upper=r_rot_control[0],
                                                                                  weight=weight,
                                                                                  expression=c_aa[0])
        soft_constraints[u'rotation 1: {}/{}'.format(root, tip)] = SoftConstraint(lower=r_rot_control[1],
                                                                                  upper=r_rot_control[1],
                                                                                  weight=weight,
                                                                                  expression=c_aa[1])
        soft_constraints[u'rotation 2: {}/{}'.format(root, tip)] = SoftConstraint(lower=r_rot_control[2],
                                                                                  upper=r_rot_control[2],
                                                                                  weight=weight,
                                                                                  expression=c_aa[2])
        return soft_constraints


class CartesianOrientationSlerp(CartesianConstraint):

    def get_constraint(self, root, tip, **kwargs):
        """
        example:
        name='CartesianPosition'
        parameter_value_pair='{
            "root": "base_footprint", #required
            "tip": "r_gripper_tool_frame", #required
            "goal_position": {"header":
                                {"stamp":
                                    {"secs": 0,
                                    "nsecs": 0},
                                "frame_id": "",
                                "seq": 0},
                            "pose": {"position":
                                        {"y": 0.0,
                                        "x": 0.0,
                                        "z": 0.0},
                                    "orientation": {"y": 0.0,
                                                    "x": 0.0,
                                                    "z": 0.0,
                                                    "w": 0.0}
                                    }
                            }', #required
            "weight": 1, #optional
            "gain": 3, #optional -- error is multiplied with this value
            "max_speed": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :param root:
        :param tip:
        :param goal_position:
        :param weights:
        :param gain:
        :param max_speed:
        :param ns:
        :return:
        """
        goal_rotation = sw.rotation_of(self.get_goal_pose(u'goal'))
        weight = self.get_symbol(u'weight')
        gain = self.get_symbol(u'gain')
        max_speed = self.get_symbol(u'max_speed')

        current_rotation = sw.rotation_of(self.get_fk(root, tip))
        current_evaluated_rotation = sw.rotation_of(self.get_fk_evaluated(root, tip))

        soft_constraints = OrderedDict()

        axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * goal_rotation))
        angle = sw.diffable_abs(angle)

        capped_angle = sw.diffable_min_fast(max_speed / (gain * angle), 1)

        q1 = sw.quaternion_from_matrix(current_rotation)
        q2 = sw.quaternion_from_matrix(goal_rotation)
        intermediate_goal = sw.diffable_slerp(q1, q2, capped_angle)
        axis, angle = sw.axis_angle_from_quaternion(*sw.quaternion_diff(q1, intermediate_goal))
        r_rot_control = axis * angle

        hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
        axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
        c_aa = (axis * angle)

        soft_constraints[str(self) + u'/1'] = SoftConstraint(lower=r_rot_control[0],
                                                             upper=r_rot_control[0],
                                                             weight=weight,
                                                             expression=c_aa[0])
        soft_constraints[str(self) + u'/2'] = SoftConstraint(lower=r_rot_control[1],
                                                             upper=r_rot_control[1],
                                                             weight=weight,
                                                             expression=c_aa[1])
        soft_constraints[str(self) + u'/3'] = SoftConstraint(lower=r_rot_control[2],
                                                             upper=r_rot_control[2],
                                                             weight=weight,
                                                             expression=c_aa[2])
        return soft_constraints


class LinkToAnyAvoidance(Constraint):
    def get_constraint(self, link_name, lower_limit=0.05, upper_limit=1e9, weight=10000):
        current_pose = self.get_fk(u'base_footprint', link_name)
        current_pose_eval = self.get_fk_evaluated(u'base_footprint', link_name)
        point_on_link = self.get_closest_point_on_a(u'robot', link_name, )
        other_point = self.get_closest_point_on_b(u'robot', link_name, )
        contact_normal = self.get_contact_normal_on_b(u'robot', link_name, )

        soft_constraints = OrderedDict()
        name = u'dist to any: {}'.format(link_name)

        controllable_point = current_pose * sw.inverse_frame(current_pose_eval) * point_on_link

        dist = (contact_normal.T * (controllable_point - other_point))[0]

        soft_constraints[u'{} '.format(name)] = SoftConstraint(lower=lower_limit - dist,
                                                               upper=upper_limit,
                                                               weight=weight,
                                                               expression=dist)
        return soft_constraints


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


def continuous_joint_position(current_joint, joint_goal, weight, p_gain, max_speed, constraint_name):
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

    err = sw.shortest_angular_distance(current_joint, joint_goal)
    capped_err = sw.diffable_max_fast(sw.diffable_min_fast(p_gain * err, max_speed), -max_speed)

    soft_constraints[constraint_name] = SoftConstraint(lower=capped_err,
                                                       upper=capped_err,
                                                       weight=weight,
                                                       expression=current_joint)
    # add_debug_constraint(soft_constraints, '{} //change//'.format(constraint_name), err)
    # add_debug_constraint(soft_constraints, '{} //curr//'.format(constraint_name), current_joint)
    # add_debug_constraint(soft_constraints, '{} //goal//'.format(constraint_name), joint_goal)
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

    soft_constraints[u'position x: {}'.format(ns)] = SoftConstraint(lower=trans_control[0],
                                                                    upper=trans_control[0],
                                                                    weight=weights,
                                                                    expression=current_position[0])
    soft_constraints[u'position y: {}'.format(ns)] = SoftConstraint(lower=trans_control[1],
                                                                    upper=trans_control[1],
                                                                    weight=weights,
                                                                    expression=current_position[1])
    soft_constraints[u'position z: {}'.format(ns)] = SoftConstraint(lower=trans_control[2],
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
    axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * goal_rotation))

    capped_angle = sw.diffable_max_fast(sw.diffable_min_fast(rot_gain * angle, max_rot_speed), -max_rot_speed)

    r_rot_control = axis * capped_angle

    hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

    axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)

    soft_constraints[u'rotation 0: {}'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                    upper=r_rot_control[0],
                                                                    weight=weights,
                                                                    expression=c_aa[0])
    soft_constraints[u'rotation 1: {}'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                    upper=r_rot_control[1],
                                                                    weight=weights,
                                                                    expression=c_aa[1])
    soft_constraints[u'rotation 2: {}'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
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

    axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * goal_rotation))
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
    axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
    c_aa = (axis * angle)
    # c_aa = current_evaluated_rotation[:3, :3] * c_aa

    soft_constraints[u'rotation 0: {}'.format(ns)] = SoftConstraint(lower=r_rot_control[0],
                                                                    upper=r_rot_control[0],
                                                                    weight=weights,
                                                                    expression=c_aa[0])
    soft_constraints[u'rotation 1: {}'.format(ns)] = SoftConstraint(lower=r_rot_control[1],
                                                                    upper=r_rot_control[1],
                                                                    weight=weights,
                                                                    expression=c_aa[1])
    soft_constraints[u'rotation 2: {}'.format(ns)] = SoftConstraint(lower=r_rot_control[2],
                                                                    upper=r_rot_control[2],
                                                                    weight=weights,
                                                                    expression=c_aa[2])
    # add_debug_constraint(soft_constraints, 'q2', q2)
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
    name = u'dist to any: {}'.format(link_name)

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
