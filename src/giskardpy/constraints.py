from __future__ import division

import numbers
import numpy as np
from collections import OrderedDict

import PyKDL as kdl
from geometry_msgs.msg import Vector3Stamped, Vector3
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message
from scipy.optimize import curve_fit

import giskardpy.identifier as identifier
import symengine_wrappers as sw
from giskardpy.exceptions import GiskardException
from giskardpy.input_system import PoseStampedInput, Point3Input, Vector3Input, Vector3StampedInput, FrameInput, \
    PointStampedInput, WrenchInput
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.tfwrapper import transform_pose, transform_vector, transform_point, to_np, np_to_kdl

MAX_WEIGHT = 15
HIGH_WEIGHT = 5
MID_WEIGHT = 1
LOW_WEIGHT = 0.5
ZERO_WEIGHT = 0


class Constraint(object):
    def __init__(self, god_map, **kwargs):
        self.god_map = god_map

    def save_params_on_god_map(self, params):
        constraints = self.get_god_map().safe_get_data(identifier.constraints_identifier)
        constraints[str(self)] = params
        self.get_god_map().safe_set_data(identifier.constraints_identifier, constraints)

    def get_constraint(self):
        pass

    def get_identifier(self):
        return identifier.constraints_identifier + [str(self)]

    def get_world_object_pose(self, object_name, link_name):
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
        return self.get_god_map().safe_get_data(identifier.world)

    def get_robot(self):
        """
        :rtype: giskardpy.symengine_robot.Robot
        """
        return self.get_god_map().safe_get_data(identifier.robot)

    def get_world_unsafe(self):
        """
        :rtype: giskardpy.world.World
        """
        return self.get_god_map().get_data(identifier.world)

    def get_robot_unsafe(self):
        """
        :rtype: giskardpy.symengine_robot.Robot
        """
        return self.get_god_map().get_data(identifier.robot)

    def get_input_joint_position(self, joint_name):
        key = identifier.joint_states + [joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_input_sampling_period(self):
        key = identifier.rosparam + [u'sample_period']
        return self.god_map.to_symbol(key)

    def __str__(self):
        return self.__class__.__name__

    def get_fk(self, root, tip):
        return self.get_robot().get_fk_expression(root, tip)

    def get_fk_evaluated(self, root, tip):
        return FrameInput(self.get_god_map().to_symbol,
                          prefix=identifier.fk_np +
                                 [(root, tip)]).get_frame()

    def get_input_float(self, name):
        key = self.get_identifier() + [name]
        return self.god_map.to_symbol(key)

    def get_input_PoseStamped(self, name):
        return PoseStampedInput(self.get_god_map().to_symbol,
                                translation_prefix=self.get_identifier() +
                                                   [name,
                                                    u'pose',
                                                    u'position'],
                                rotation_prefix=self.get_identifier() +
                                                [name,
                                                 u'pose',
                                                 u'orientation']).get_frame()

    def get_input_Vector3Stamped(self, name):
        return Vector3StampedInput(self.god_map.to_symbol,
                                   vector_prefix=self.get_identifier() + [name, u'vector']).get_expression()

    def get_input_PointStamped(self, name):
        return PointStampedInput(self.god_map.to_symbol,
                                 prefix=self.get_identifier() + [name, u'point']).get_expression()


class JointPosition(Constraint):
    goal = u'goal'
    weight = u'weight'
    gain = u'gain'
    max_speed = u'max_speed'

    def __init__(self, god_map, joint_name, goal, weight=LOW_WEIGHT, gain=1, max_speed=1):
        super(JointPosition, self).__init__(god_map)
        self.joint_name = joint_name

        params = {self.goal: goal,
                  self.weight: weight,
                  self.gain: gain,
                  self.max_speed: max_speed}
        self.save_params_on_god_map(params)

    def get_constraint(self):
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
        :return:
        """
        current_joint = self.get_input_joint_position(self.joint_name)

        joint_goal = self.get_input_float(self.goal)
        weight = self.get_input_float(self.weight)
        # p_gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)
        t = self.get_input_sampling_period()

        soft_constraints = OrderedDict()

        if self.get_robot().is_joint_continuous(self.joint_name):
            err = sw.shortest_angular_distance(current_joint, joint_goal)
        else:
            err = joint_goal - current_joint
        capped_err = sw.diffable_max_fast(sw.diffable_min_fast(err, max_speed * t), -max_speed * t)

        soft_constraints[str(self)] = SoftConstraint(lower=capped_err,
                                                     upper=capped_err,
                                                     weight=weight,
                                                     expression=current_joint)
        return soft_constraints

    def __str__(self):
        s = super(JointPosition, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionList(Constraint):
    def __init__(self, god_map, goal_state, weight=None, gain=None, max_speed=None):
        super(JointPositionList, self).__init__(god_map)
        self.constraints = []
        for i, joint_name in enumerate(goal_state[u'name']):
            goal_position = goal_state[u'position'][i]
            params = {u'joint_name': joint_name,
                      u'goal': goal_position}
            if weight is not None:
                params[u'weight'] = weight
            if gain is not None:
                params[u'gain'] = gain
            if max_speed is not None:
                params[u'max_speed'] = max_speed
            self.constraints.append(JointPosition(god_map, **params))

    def get_constraint(self):
        soft_constraints = OrderedDict()
        for constraint in self.constraints:
            soft_constraints.update(constraint.get_constraint())
        return soft_constraints


class CartesianConstraint(Constraint):
    goal = u'goal'
    weight = u'weight'
    gain = u'gain'
    max_speed = u'max_speed'

    def __init__(self, god_map, root_link, tip_link, goal, weight=HIGH_WEIGHT, gain=1, max_speed=0.1):
        super(CartesianConstraint, self).__init__(god_map)
        self.root = root_link
        self.tip = tip_link
        goal = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped', goal)
        goal = transform_pose(self.root, goal)

        # make sure rotation is normalized quaternion
        # TODO make a function out of this
        rotation = np.array([goal.pose.orientation.x,
                             goal.pose.orientation.y,
                             goal.pose.orientation.z,
                             goal.pose.orientation.w])
        normalized_rotation = rotation / np.linalg.norm(rotation)
        goal.pose.orientation.x = normalized_rotation[0]
        goal.pose.orientation.y = normalized_rotation[1]
        goal.pose.orientation.z = normalized_rotation[2]
        goal.pose.orientation.w = normalized_rotation[3]

        params = {self.goal: goal,
                  self.weight: weight,
                  self.gain: gain,
                  self.max_speed: max_speed}
        self.save_params_on_god_map(params)

    def get_goal_pose(self):
        return self.get_input_PoseStamped(self.goal)

    def __str__(self):
        s = super(CartesianConstraint, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class CartesianPosition(CartesianConstraint):

    def get_constraint(self):
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
        :return:
        """

        goal_position = sw.position_of(self.get_goal_pose())
        weight = self.get_input_float(self.weight)
        gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)
        t = self.get_input_sampling_period()

        current_position = sw.position_of(self.get_fk(self.root, self.tip))

        soft_constraints = OrderedDict()

        trans_error_vector = goal_position - current_position
        trans_error = sw.norm(trans_error_vector)
        trans_scale = sw.diffable_min_fast(trans_error * gain, max_speed * t)
        trans_control = sw.save_division(trans_error_vector, trans_error) * trans_scale

        soft_constraints[str(self) + u'x'] = SoftConstraint(lower=trans_control[0],
                                                            upper=trans_control[0],
                                                            weight=weight,
                                                            expression=current_position[0])
        soft_constraints[str(self) + u'y'] = SoftConstraint(lower=trans_control[1],
                                                            upper=trans_control[1],
                                                            weight=weight,
                                                            expression=current_position[1])
        soft_constraints[str(self) + u'z'] = SoftConstraint(lower=trans_control[2],
                                                            upper=trans_control[2],
                                                            weight=weight,
                                                            expression=current_position[2])
        return soft_constraints


class CartesianPositionX(CartesianConstraint):
    def get_constraint(self):
        goal_position = sw.position_of(self.get_goal_pose())
        weight = self.get_input_float(self.weight)
        gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)
        t = self.get_input_sampling_period()

        current_position = sw.position_of(self.get_fk(self.root, self.tip))

        soft_constraints = OrderedDict()

        trans_error_vector = goal_position - current_position
        trans_error = sw.norm(trans_error_vector)
        trans_scale = sw.diffable_min_fast(trans_error * gain, max_speed * t)
        trans_control = sw.save_division(trans_error_vector, trans_error) * trans_scale

        soft_constraints[str(self) + u'x'] = SoftConstraint(lower=trans_control[0],
                                                            upper=trans_control[0],
                                                            weight=weight,
                                                            expression=current_position[0])
        return soft_constraints


class CartesianPositionY(CartesianConstraint):
    def get_constraint(self):
        goal_position = sw.position_of(self.get_goal_pose())
        weight = self.get_input_float(self.weight)
        gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)
        t = self.get_input_sampling_period()

        current_position = sw.position_of(self.get_fk(self.root, self.tip))

        soft_constraints = OrderedDict()

        trans_error_vector = goal_position - current_position
        trans_error = sw.norm(trans_error_vector)
        trans_scale = sw.diffable_min_fast(trans_error * gain, max_speed * t)
        trans_control = sw.save_division(trans_error_vector, trans_error) * trans_scale

        soft_constraints[str(self) + u'y'] = SoftConstraint(lower=trans_control[1],
                                                            upper=trans_control[1],
                                                            weight=weight,
                                                            expression=current_position[1])
        return soft_constraints


class CartesianOrientation(CartesianConstraint):
    def __init__(self, god_map, root_link, tip_link, goal, weight=HIGH_WEIGHT, gain=1, max_speed=0.5):
        super(CartesianOrientation, self).__init__(god_map, root_link, tip_link, goal, weight, gain, max_speed)

    def get_constraint(self):
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
        :return:
        """
        goal_rotation = sw.rotation_of(self.get_goal_pose())
        weight = self.get_input_float(self.weight)
        gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)

        current_rotation = sw.rotation_of(self.get_fk(self.root, self.tip))
        current_evaluated_rotation = sw.rotation_of(self.get_fk_evaluated(self.root, self.tip))

        soft_constraints = OrderedDict()
        axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * goal_rotation))

        capped_angle = sw.diffable_max_fast(sw.diffable_min_fast(gain * angle, max_speed), -max_speed)

        r_rot_control = axis * capped_angle

        hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

        axis, angle = sw.diffable_axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
        c_aa = (axis * angle)

        soft_constraints[str(self) + u'/0'] = SoftConstraint(lower=r_rot_control[0],
                                                             upper=r_rot_control[0],
                                                             weight=weight,
                                                             expression=c_aa[0])
        soft_constraints[str(self) + u'/1'] = SoftConstraint(lower=r_rot_control[1],
                                                             upper=r_rot_control[1],
                                                             weight=weight,
                                                             expression=c_aa[1])
        soft_constraints[str(self) + u'/2'] = SoftConstraint(lower=r_rot_control[2],
                                                             upper=r_rot_control[2],
                                                             weight=weight,
                                                             expression=c_aa[2])
        return soft_constraints


class CartesianOrientationSlerp(CartesianConstraint):
    def __init__(self, god_map, root_link, tip_link, goal, weight=HIGH_WEIGHT, gain=1, max_speed=1):
        super(CartesianOrientationSlerp, self).__init__(god_map, root_link, tip_link, goal, weight, gain, max_speed)

    def get_constraint(self):
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
        :return:
        """
        goal_rotation = sw.rotation_of(self.get_goal_pose())
        weight = self.get_input_float(self.weight)
        gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)
        t = self.get_input_sampling_period()

        current_rotation = sw.rotation_of(self.get_fk(self.root, self.tip))
        current_evaluated_rotation = sw.rotation_of(self.get_fk_evaluated(self.root, self.tip))

        soft_constraints = OrderedDict()

        angle = sw.rotation_distance(current_rotation, goal_rotation)
        angle = sw.diffable_abs(angle)

        capped_angle = sw.diffable_min_fast(sw.save_division(max_speed, (gain * angle)), 1)
        q1 = sw.quaternion_from_matrix(current_rotation)
        q2 = sw.quaternion_from_matrix(goal_rotation)
        intermediate_goal = sw.diffable_slerp(q1, q2, capped_angle)
        asdf = sw.quaternion_diff(q1, intermediate_goal)
        axis3, angle3 = sw.axis_angle_from_quaternion(*asdf)
        r_rot_control = axis3 * angle3

        hack = sw.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
        axis2, angle2 = sw.diffable_axis_angle_from_matrix((current_rotation.T * (current_evaluated_rotation * hack)).T)
        c_aa = (axis2 * angle2)

        soft_constraints[str(self) + u'/0'] = SoftConstraint(lower=r_rot_control[0],
                                                             upper=r_rot_control[0],
                                                             weight=weight,
                                                             expression=c_aa[0])
        soft_constraints[str(self) + u'/1'] = SoftConstraint(lower=r_rot_control[1],
                                                             upper=r_rot_control[1],
                                                             weight=weight,
                                                             expression=c_aa[1])
        soft_constraints[str(self) + u'/2'] = SoftConstraint(lower=r_rot_control[2],
                                                             upper=r_rot_control[2],
                                                             weight=weight,
                                                             expression=c_aa[2])
        return soft_constraints


class LinkToClosestAvoidance(Constraint):
    repel_speed = u'repel_speed'
    max_weight_distance = u'max_weight_distance'
    low_weight_distance = u'low_weight_distance'
    zero_weight_distance = u'zero_weight_distance'
    root_T_link_b = u'root_T_link_b'
    link_in_chain = u'link_in_chain'
    A = u'A'
    B = u'B'
    C = u'C'

    def __init__(self, god_map, link_name, repel_speed=0.5, max_weight_distance=0.0, low_weight_distance=0.01,
                 zero_weight_distance=0.05, idx=0):
        super(LinkToClosestAvoidance, self).__init__(god_map)
        self.link_name = link_name
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()
        self.idx = idx
        x = np.array([max_weight_distance, low_weight_distance, zero_weight_distance])
        y = np.array([MAX_WEIGHT, LOW_WEIGHT, ZERO_WEIGHT])
        (A, B, C), _ = curve_fit(lambda t, a, b, c: a / (t + c) + b, x, y)

        identity = np.eye(4)

        cpi_identifier = identifier.closest_point[0]
        world_identifier = identifier.world[0]
        data = god_map._data
        robot = data[world_identifier].robot
        get_fk = robot.get_fk_np
        get_chain = robot.get_chain
        get_connecting_link = robot.get_connecting_link

        # TODO rename me
        def root_T_link_b():
            try:
                cpi = data[cpi_identifier][(self.link_name,)][self.idx]
                if cpi.body_b == self.robot_name:
                    c = get_connecting_link(cpi.link_b, cpi.link_a)
                    return get_fk(self.robot_root, c)
            except IndexError:
                pass
            return identity

        def link_in_chain(link_name):
            try:
                cpi = data[cpi_identifier][(self.link_name,)][self.idx]
                body_b = cpi.body_b
                if body_b == self.robot_name:
                    link_a = cpi.link_a
                    link_b = cpi.link_b
                    chain = get_chain(link_b, link_a, joints=False)
                    return int(link_name in chain)
            except IndexError:
                pass
            return 1

        params = {self.repel_speed: repel_speed,
                  self.max_weight_distance: max_weight_distance,
                  self.low_weight_distance: low_weight_distance,
                  self.zero_weight_distance: zero_weight_distance,
                  self.link_in_chain: link_in_chain,
                  self.root_T_link_b: root_T_link_b,
                  self.A: A,
                  self.B: B,
                  self.C: C, }
        self.save_params_on_god_map(params)

    def get_distance_to_closest_object(self, link_name):
        return self.get_god_map().to_symbol(identifier.closest_point + [(link_name,), self.idx, u'min_dist'])

    def get_contact_normal_on_b(self, link_name):
        return Vector3Input(self.god_map.to_symbol,
                            prefix=identifier.closest_point + [(link_name,), self.idx,
                                                               u'contact_normal']).get_expression()

    def get_closest_point_on_a(self, link_name):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [(link_name,), self.idx,
                                                              u'position_on_a']).get_expression()

    def get_closest_point_on_b(self, link_name):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [(link_name,), self.idx,
                                                              u'position_on_b']).get_expression()

    def get_actual_distance(self, link_name):
        return self.god_map.to_symbol(identifier.closest_point + [(link_name,), self.idx, u'contact_distance'])

    def get_is_link_in_chain_symbol(self, link_name):
        return self.god_map.to_symbol(self.get_identifier() + [self.link_in_chain, (link_name,)])

    def get_root_T_link_b(self):
        return FrameInput(self.god_map.to_symbol, self.get_identifier() + [self.root_T_link_b, tuple()]).get_frame()

    def get_constraint(self):
        soft_constraints = OrderedDict()

        root_T_link = self.get_root_T_link_b()
        chain = self.get_robot().get_chain(self.get_robot().get_root(), self.link_name, False, True, False)
        for i in range(len(chain) - 1):
            l1 = chain[i]
            l2 = chain[i + 1]
            link_in_chain = self.get_is_link_in_chain_symbol(l1)
            fk_expr = self.get_fk(l1, l2)
            root_T_link *= sw.if_eq_zero(link_in_chain, sw.eye(4), fk_expr)
            # add_debug_constraint(soft_constraints, str(self)+'/link in chain / '+l1, link_in_chain)

        point_on_link = self.get_closest_point_on_a(self.link_name)
        # other_point = self.get_closest_point_on_b(self.link_name)
        contact_normal = self.get_contact_normal_on_b(self.link_name)
        actual_distance = self.get_actual_distance(self.link_name)
        repel_speed = self.get_input_float(self.repel_speed)
        t = self.get_input_sampling_period()
        max_weight_distance = self.get_input_float(self.max_weight_distance)
        zero_weight_distance = self.get_input_float(self.zero_weight_distance)
        A = self.get_input_float(self.A)
        B = self.get_input_float(self.B)
        C = self.get_input_float(self.C)

        controllable_point = root_T_link * point_on_link

        dist = (contact_normal.T * (controllable_point))[0]

        # weight_f = sw.Piecewise([MAX_WEIGHT, actual_distance <= max_weight_distance],
        #                         [ZERO_WEIGHT, actual_distance > zero_weight_distance],
        #                         [A / (actual_distance + C) + B, True])

        weight_f = sw.Max(sw.Min(MAX_WEIGHT, A / (actual_distance + C) + B), ZERO_WEIGHT)

        limit = zero_weight_distance - actual_distance
        limit = sw.Min(sw.Max(limit, -repel_speed * t), repel_speed * t)
        # sw.Min(, repel_speed * t)
        # limit = repel_speed * t
        # limit = 1

        soft_constraints[str(self)] = SoftConstraint(lower=limit,
                                                     upper=1e9,
                                                     weight=weight_f,
                                                     expression=dist)
        return soft_constraints

    def __str__(self):
        s = super(LinkToClosestAvoidance, self).__str__()
        return u'{}/{}/{}'.format(s, self.link_name, self.idx)


class EMAAvoidance(Constraint):
    repel_speed = u'repel_speed'
    max_weight_distance = u'max_weight_distance'
    low_weight_distance = u'low_weight_distance'
    zero_weight_distance = u'zero_weight_distance'
    root_T_link_b = u'root_T_link_b'
    link_in_chain = u'link_in_chain'
    max_speed = u'max_speed'
    A = u'A'
    B = u'B'
    C = u'C'
    ema_normal = u'ema_normal'
    weights = u'weights'
    a = u'a'
    b = u'b'

    def __init__(self, god_map, link_name, repel_speed=0.1, max_weight_distance=0.0, low_weight_distance=0.01,
                 zero_weight_distance=0.05, idx=0, max_speed=0.1):
        super(EMAAvoidance, self).__init__(god_map)
        self.link_name = link_name
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()
        self.idx = idx
        sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        # self.gamma = 0.5 ** (sample_period)
        # self.gamma = int(1/sample_period)*10
        self.gamma = 1
        # x = np.array([max_weight_distance, low_weight_distance, zero_weight_distance])
        # y = np.array([ZERO_WEIGHT, LOW_WEIGHT, MAX_WEIGHT])
        # (A, B, C), _ = curve_fit(lambda t, a, b, c: -a / (t - c) + b, x, y)
        x = np.array([max_weight_distance, zero_weight_distance])
        y = np.array([ZERO_WEIGHT, MAX_WEIGHT])
        (A, B), _ = curve_fit(lambda t, a, b: (t * a) + b, x, y)

        identity = np.eye(4)

        cpi_identifier = identifier.closest_point[0]
        world_identifier = identifier.world[0]
        data = god_map._data
        robot = data[world_identifier].robot
        get_fk = robot.get_fk_np
        get_chain = robot.get_chain
        get_connecting_link = robot.get_connecting_link

        # TODO rename me
        def root_T_link_b():
            try:
                cpi = data[cpi_identifier][(self.link_name,)][self.idx]
                if cpi.body_b == self.robot_name:
                    c = get_connecting_link(cpi.link_b, cpi.link_a)
                    return get_fk(self.robot_root, c)
            except IndexError:
                pass
            return identity

        def link_in_chain(link_name):
            try:
                cpi = data[cpi_identifier][(self.link_name,)][self.idx]
                body_b = cpi.body_b
                if body_b == self.robot_name:
                    link_a = cpi.link_a
                    link_b = cpi.link_b
                    chain = get_chain(link_b, link_a, joints=False)
                    return int(link_name in chain)
            except IndexError:
                pass
            return 1

        class EMA(object):
            def __init__(self2, f, gamma, link_name, normalize=False):
                self2.f = f
                self2.gamma = gamma
                self2.link_name = link_name
                self2.last_value = None
                self2.normalize = normalize

            def __call__(self2):
                normals = []
                weights = []
                for collision in data[cpi_identifier][(self2.link_name,)][:5]:
                    normals.append(np.array([self2.f(collision)[0],
                                             self2.f(collision)[1],
                                             self2.f(collision)[2]]))
                    weights.append(1 / (collision.contact_distance) ** 2)
                if len(normals) != 0:
                    current_normal = np.average(normals, axis=0, weights=weights)
                    if self2.last_value is None:
                        self2.last_value = current_normal
                    else:
                        self2.last_value = self2.last_value * self2.gamma + current_normal * (1 - self2.gamma)
                        if self2.normalize:
                            self2.last_value /= np.linalg.norm(self2.last_value)
                    return self2.last_value
                else:
                    return [0, 0, 0]

        class AVG(object):
            def __init__(self2, f, wl, link_name):
                self2.f = f
                self2.wl = wl
                self2.link_name = link_name
                self2.last_values = []

            def __call__(self2):
                normals = []
                weights = []
                for collision in data[cpi_identifier][(self2.link_name,)][:5]:
                    d = max(collision.min_dist - collision.contact_distance, 0)
                    normals.append(np.array([self2.f(collision)[0],
                                             self2.f(collision)[1],
                                             self2.f(collision)[2]]))
                    weights.append(d)
                if len(normals) != 0:
                    current_normal = np.average(normals, axis=0, weights=weights)
                    self2.last_values.append(current_normal)
                    self2.last_values = self2.last_values[-self2.wl:]
                    last_value = np.average(self2.last_values, axis=0)
                    return last_value
                else:
                    return [0, 0, 0]

        class AVG2(object):
            def __init__(self2, f, wl, link_name):
                self2.f = f
                self2.wl = wl
                self2.link_name = link_name
                self2.last_values = []

            def __call__(self2):
                vs = []
                for collision in data[cpi_identifier][(self2.link_name,)]:
                    d = max(zero_weight_distance - collision.contact_distance, 0)
                    if d == 0:
                        vs.append(np.zeros(6))
                        continue
                    link_T_root = np_to_kdl(get_fk(self2.link_name, robot.get_root()))
                    v = kdl.Vector()
                    v[0] = self2.f(collision)[0]
                    v[1] = self2.f(collision)[1]
                    v[2] = self2.f(collision)[2]
                    # if collision.contact_distance < 0:
                    #     v = -v
                    v_link = link_T_root.M * v

                    r = np.array([collision.position_on_a[0],
                                  collision.position_on_a[1],
                                  collision.position_on_a[2]])
                    f = to_np(v_link * d)
                    rr1 = r / np.linalg.norm(r)
                    wrench = np.zeros(6)
                    wrench[:3] = rr1 * np.dot(rr1, f)
                    wrench[3:] = np.cross(r, f - wrench[:3])
                    vs.append(wrench)

                if len(vs) != 0:
                    vs = np.atleast_2d(vs).T
                    current_normal = np.sum(vs, axis=1)
                    self2.last_values.append(current_normal)
                    self2.last_values = self2.last_values[-self2.wl:]
                    last_value = np.average(self2.last_values, axis=0)
                    return last_value
                else:
                    return [0, 0, 0]

        class Weights(object):
            def __init__(self2, link_name):
                self2.link_name = link_name
                self2.last_values = []

            def __call__(self2):
                vs = []
                for collision in data[cpi_identifier][(self2.link_name,)]:
                    d = max(zero_weight_distance - collision.contact_distance, 0)
                    if d == 0:
                        # vs.append(np.zeros(6))
                        continue
                    link_T_root = np_to_kdl(get_fk(self2.link_name, robot.get_root()))
                    v = kdl.Vector()
                    v[0] = collision.contact_normal[0]
                    v[1] = collision.contact_normal[1]
                    v[2] = collision.contact_normal[2]
                    # if collision.contact_distance < 0:
                    #     v = -v
                    v_link = link_T_root.M * v

                    r = np.array([collision.position_on_a[0],
                                  collision.position_on_a[1],
                                  collision.position_on_a[2]])
                    f = to_np(v_link * d)
                    rr1 = r / np.linalg.norm(r)
                    wrench = np.zeros(6)
                    wrench[:3] = rr1 * np.dot(rr1, f)
                    wrench[3:] = np.cross(r, f - wrench[:3])
                    vs.append(wrench)

                if len(vs) != 0:
                    vs = np.atleast_2d(vs).T
                    # todo ignore 0 vectors
                    vs = np.abs(vs)
                    current_normal = np.sum(vs, axis=1)
                    # self2.last_values.append(current_normal)
                    # last_value = np.average(self2.last_values, axis=0)
                    return current_normal
                else:
                    return [0, 0, 0]

        params = {self.repel_speed: repel_speed,
                  self.max_weight_distance: max_weight_distance,
                  self.low_weight_distance: low_weight_distance,
                  self.zero_weight_distance: zero_weight_distance,
                  self.max_speed: max_speed,
                  self.link_in_chain: link_in_chain,
                  self.root_T_link_b: root_T_link_b,
                  self.ema_normal: AVG2(lambda x: x.contact_normal, self.gamma, self.link_name),
                  self.a: AVG(lambda x: x.position_on_a, self.gamma, self.link_name),
                  self.b: AVG(lambda x: x.position_on_b, self.gamma, self.link_name),
                  self.weights: Weights(self.link_name),
                  self.A: A,
                  self.B: B,
                  self.C: 0}
        self.save_params_on_god_map(params)

    def get_distance_to_closest_object(self, link_name):
        return self.get_god_map().to_symbol(identifier.closest_point + [(link_name,), self.idx, u'min_dist'])

    def get_contact_normal_on_b(self):
        return WrenchInput(self.god_map.to_symbol,
                           self.get_identifier() + [self.ema_normal, tuple()]).get_expression()

    def get_weights(self):
        return WrenchInput(self.god_map.to_symbol,
                           self.get_identifier() + [self.weights, tuple()]).get_expression()

    def get_closest_point_on_a(self):
        return Point3Input(self.god_map.to_symbol,
                           self.get_identifier() + [self.a, tuple()]).get_expression()

    def get_closest_point_on_b(self):
        return Point3Input(self.god_map.to_symbol,
                           self.get_identifier() + [self.b, tuple()]).get_expression()

    def get_actual_distance(self, link_name):
        return self.god_map.to_symbol(identifier.closest_point + [(link_name,), self.idx, u'contact_distance'])

    def get_is_link_in_chain_symbol(self, link_name):
        return self.god_map.to_symbol(self.get_identifier() + [self.link_in_chain, (link_name,)])

    def get_root_T_link_b(self):
        return FrameInput(self.god_map.to_symbol, self.get_identifier() + [self.root_T_link_b, tuple()]).get_frame()

    def get_constraint(self):
        soft_constraints = OrderedDict()

        root_T_link = self.get_root_T_link_b()
        chain = self.get_robot().get_chain(self.get_robot().get_root(), self.link_name, False, True, False)
        for i in range(len(chain) - 1):
            l1 = chain[i]
            l2 = chain[i + 1]
            link_in_chain = self.get_is_link_in_chain_symbol(l1)
            fk_expr = self.get_fk(l1, l2)
            root_T_link *= sw.if_eq_zero(link_in_chain, sw.eye(4), fk_expr)

        # point_on_link = self.get_closest_point_on_a()
        # other_point = self.get_closest_point_on_b()
        contact_normal = self.get_contact_normal_on_b()
        actual_distance = self.get_actual_distance(self.link_name)
        # repel_speed = self.get_input_float(self.repel_speed)
        max_speed = self.get_input_float(self.max_speed)
        t = self.get_input_sampling_period()
        max_weight_distance = self.get_input_float(self.max_weight_distance)
        zero_weight_distance = self.get_input_float(self.zero_weight_distance)
        A = self.get_input_float(self.A)
        B = self.get_input_float(self.B)
        C = self.get_input_float(self.C)
        weights = self.get_weights()

        # controllable_point = root_T_link * point_on_link

        # dist = (contact_normal.T * (controllable_point - other_point))[0]

        w11 = root_T_link * sw.vector3(*weights[:3])
        w11 = sw.Matrix([sw.Abs(w) for w in w11])
        # w21 = root_T_link * sw.vector3(*weights[3:])
        w21 = sw.Matrix([sw.Abs(w) for w in weights[3:]])
        w1_scale = sw.norm(w11)
        w2_scale = sw.norm(w21)
        w1_f = sw.Max(ZERO_WEIGHT, sw.Min(MAX_WEIGHT, 2.5 * (w1_scale * A) + B))
        w2_f = sw.Max(ZERO_WEIGHT, sw.Min(MAX_WEIGHT, 5 * (w2_scale * A) + B))
        w1 = sw.scale(w11, w1_f)
        w2 = sw.scale(w21, w2_f)

        # limit = sw.Min(zero_weight_distance - dist, repel_speed * t)
        # contact_normal *= sw.Max(0, sw.Min(zero_weight_distance - actual_distance, repel_speed * t))
        # limit = repel_speed * t
        # limit = 1

        trans_error_vector = sw.Matrix(contact_normal[:3])
        trans_error = sw.norm(trans_error_vector)
        trans_scale = sw.diffable_min_fast(trans_error, max_speed * t)
        trans_control = self.get_fk(self.get_robot().get_root(), self.link_name) * \
                        sw.vector3(*sw.save_division(trans_error_vector, trans_error) * trans_scale)
        current_position = sw.position_of(root_T_link)

        soft_constraints[str(self) + u'/fx'] = SoftConstraint(lower=trans_control[0],
                                                              upper=trans_control[0],
                                                              weight=w1[0],
                                                              expression=current_position[0])
        # soft_constraints[str(self) + u'/fy'] = SoftConstraint(lower=trans_control[1],
        #                                                       upper=trans_control[1],
        #                                                       weight=w1[1],
        #                                                       expression=current_position[1])
        # soft_constraints[str(self) + u'/fz'] = SoftConstraint(lower=trans_control[2],
        #                                                       upper=trans_control[2],
        #                                                       weight=w1[2],
        #                                                       expression=current_position[2])
        # add_debug_constraint(soft_constraints, str(self)+'/w1_scale', w1_scale)
        # add_debug_constraint(soft_constraints, str(self)+'/w2_scale', w2_scale)
        # add_debug_constraint(soft_constraints, str(self)+'/w1_f', w1_f)
        # add_debug_constraint(soft_constraints, str(self)+'/w2_f', w2_f)
        #
        #
        # add_debug_constraint(soft_constraints, str(self)+'/w11/x', w11[0])
        # add_debug_constraint(soft_constraints, str(self)+'/w11/y', w11[1])
        # add_debug_constraint(soft_constraints, str(self)+'/w11/z', w11[2])
        #
        # add_debug_constraint(soft_constraints, str(self)+'/b/x', other_point[0])
        # add_debug_constraint(soft_constraints, str(self)+'/b/y', other_point[1])
        # add_debug_constraint(soft_constraints, str(self)+'/b/z', other_point[2])

        # goal_rotation = sw.Matrix(contact_normal[3:])
        # # weight = self.get_input_float(self.weight)
        # max_speed = self.get_input_float(self.max_speed)
        # t = self.get_input_sampling_period()
        #
        # # root_R_link = sw.rotation_of(self.get_fk(self.get_robot().get_root(), self.link_name))
        # root_R_link_eval = sw.rotation_of(self.get_fk_evaluated(self.get_robot().get_root(), self.link_name))
        #
        # angle = sw.norm(goal_rotation)
        # angle = sw.diffable_abs(angle)
        #
        # capped_angle = sw.diffable_min_fast(max_speed * t, angle)
        # goal_rotation = sw.save_division(goal_rotation, angle)
        # goal_rotation *= capped_angle
        #
        # hack = sw.rotation_matrix_from_axis_angle([0, 1, 0], 0.0001)
        # axis2, angle2 = sw.diffable_axis_angle_from_matrix(
        #     (sw.rotation_of(root_T_link).T * (root_R_link_eval * hack)).T)
        # c_aa = (axis2 * angle2)
        #
        # soft_constraints[str(self) + u'/tx'] = SoftConstraint(lower=goal_rotation[0],
        #                                                       upper=goal_rotation[0],
        #                                                       weight=w2[0],
        #                                                       expression=c_aa[0])
        # soft_constraints[str(self) + u'/ty'] = SoftConstraint(lower=goal_rotation[1],
        #                                                       upper=goal_rotation[1],
        #                                                       weight=w2[1],
        #                                                       expression=c_aa[1])
        # soft_constraints[str(self) + u'/tz'] = SoftConstraint(lower=goal_rotation[2],
        #                                                       upper=goal_rotation[2],
        #                                                       weight=w2[2],
        #                                                       expression=c_aa[2])

        return soft_constraints

    def __str__(self):
        s = super(EMAAvoidance, self).__str__()
        return u'{}/{}/{}'.format(s, self.link_name, self.idx)


class AlignPlanes(Constraint):
    root_normal = u'root_normal'
    tip_normal = u'tip_normal'
    weight = u'weight'
    gain = u'gain'
    max_speed = u'max_speed'

    def __init__(self, god_map, root, tip, root_normal, tip_normal, weight=HIGH_WEIGHT, gain=3, max_speed=0.5):
        """
        :type god_map:
        :type root: str
        :type tip: str
        :type root_normal: Vector3Stamped
        :type tip_normal: Vector3Stamped
        """
        super(AlignPlanes, self).__init__(god_map)
        self.root = root
        self.tip = tip

        root_normal = convert_dictionary_to_ros_message(u'geometry_msgs/Vector3Stamped', root_normal)
        root_normal = transform_vector(self.root, root_normal)
        tmp = np.array([root_normal.vector.x, root_normal.vector.y, root_normal.vector.z])
        tmp = tmp / np.linalg.norm(tmp)
        root_normal.vector = Vector3(*tmp)

        tip_normal = convert_dictionary_to_ros_message(u'geometry_msgs/Vector3Stamped', tip_normal)
        tip_normal = transform_vector(self.tip, tip_normal)
        tmp = np.array([tip_normal.vector.x, tip_normal.vector.y, tip_normal.vector.z])
        tmp = tmp / np.linalg.norm(tmp)
        tip_normal.vector = Vector3(*tmp)

        params = {self.root_normal: root_normal,
                  self.tip_normal: tip_normal,
                  self.weight: weight,
                  self.gain: gain,
                  self.max_speed: max_speed}
        self.save_params_on_god_map(params)

    def __str__(self):
        s = super(AlignPlanes, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)

    def get_root_normal_vector(self):
        return self.get_input_Vector3Stamped(self.root_normal)

    def get_tip_normal_vector(self):
        return self.get_input_Vector3Stamped(self.tip_normal)

    def get_constraint(self):
        # TODO integrate gain and max_speed?
        soft_constraints = OrderedDict()
        weight = self.get_input_float(self.weight)
        gain = self.get_input_float(self.gain)
        max_speed = self.get_input_float(self.max_speed)
        root_R_tip = sw.rotation_of(self.get_fk(self.root, self.tip))
        tip_normal__tip = self.get_tip_normal_vector()
        root_normal__root = self.get_root_normal_vector()

        tip_normal__root = root_R_tip * tip_normal__tip
        diff = root_normal__root - tip_normal__root

        soft_constraints[str(self) + u'x'] = SoftConstraint(lower=diff[0],
                                                            upper=diff[0],
                                                            weight=weight,
                                                            expression=tip_normal__root[0])
        soft_constraints[str(self) + u'y'] = SoftConstraint(lower=diff[1],
                                                            upper=diff[1],
                                                            weight=weight,
                                                            expression=tip_normal__root[1])
        soft_constraints[str(self) + u'z'] = SoftConstraint(lower=diff[2],
                                                            upper=diff[2],
                                                            weight=weight,
                                                            expression=tip_normal__root[2])
        return soft_constraints


class GravityJoint(Constraint):
    weight = u'weight'

    def __init__(self, god_map, joint_name, object_name, weight=MAX_WEIGHT):
        super(GravityJoint, self).__init__(god_map)
        self.joint_name = joint_name
        self.weight = weight
        self.object_name = object_name
        params = {self.weight: weight}
        self.save_params_on_god_map(params)

    def get_constraint(self):
        soft_constraints = OrderedDict()

        current_joint = self.get_input_joint_position(self.joint_name)
        weight = self.get_input_float(self.weight)

        parent_link = self.get_robot().get_parent_link_of_joint(self.joint_name)

        parent_R_root = sw.rotation_of(self.get_fk(parent_link, self.get_robot().get_root()))

        com__parent = sw.position_of(self.get_fk_evaluated(parent_link, self.object_name))
        com__parent[3] = 0
        com__parent = sw.scale(com__parent, 1)

        g = sw.vector3(0, 0, -1)
        g__parent = parent_R_root * g
        axis_of_rotation = sw.vector3(*self.get_robot().get_joint_axis(self.joint_name))
        l = sw.dot(g__parent, axis_of_rotation)
        goal__parent = g__parent - sw.scale(axis_of_rotation, l)
        goal__parent = sw.scale(goal__parent, 1)

        goal_vel = sw.acos(sw.dot(com__parent, goal__parent))

        ref_axis_of_rotation = sw.cross(com__parent, goal__parent)
        goal_vel *= sw.sign(sw.dot(ref_axis_of_rotation, axis_of_rotation))

        soft_constraints[str(self)] = SoftConstraint(lower=goal_vel,  # sw.Min(goal_vel, 0),
                                                     upper=goal_vel,  # sw.Max(goal_vel, 0),
                                                     weight=weight,
                                                     expression=current_joint)

        return soft_constraints

    def __str__(self):
        s = super(GravityJoint, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class UpdateGodMap(Constraint):

    def __init__(self, god_map, updates):
        super(UpdateGodMap, self).__init__(god_map)
        self.update_god_map([], updates)

    def update_god_map(self, identifier, updates):
        if not isinstance(updates, dict):
            raise GiskardException(u'{} used incorrectly, {} not a dict or number'.format(str(self), updates))
        for member, value in updates.items():
            next_identifier = identifier + [member]
            if isinstance(value, numbers.Number) and \
                    isinstance(self.get_god_map().safe_get_data(next_identifier), numbers.Number):
                self.get_god_map().safe_set_data(next_identifier, value)
            else:
                self.update_god_map(next_identifier, value)

    def get_constraint(self):
        return {}


class Pointing(Constraint):
    goal_point = u'goal_point'
    pointing_axis = u'pointing_axis'
    weight = u'weight'

    def __init__(self, god_map, tip, goal_point, root=None, pointing_axis=None, weight=HIGH_WEIGHT):
        super(Pointing, self).__init__(god_map)
        if root is None:
            self.root = self.get_robot().get_root()
        else:
            self.root = root
        self.tip = tip

        goal_point = convert_dictionary_to_ros_message(u'geometry_msgs/PointStamped', goal_point)
        goal_point = transform_point(self.root, goal_point)

        if pointing_axis is not None:
            pointing_axis = convert_dictionary_to_ros_message(u'geometry_msgs/Vector3Stamped', pointing_axis)
            pointing_axis = transform_vector(self.tip, pointing_axis)
        else:
            pointing_axis = Vector3Stamped()
            pointing_axis.header.frame_id = self.tip
            pointing_axis.vector.z = 1
        tmp = np.array([pointing_axis.vector.x, pointing_axis.vector.y, pointing_axis.vector.z])
        tmp = tmp / np.linalg.norm(tmp)  # TODO possible /0
        pointing_axis.vector = Vector3(*tmp)

        params = {self.goal_point: goal_point,
                  self.pointing_axis: pointing_axis,
                  self.weight: weight}
        self.save_params_on_god_map(params)

    def get_goal_point(self):
        return self.get_input_PointStamped(self.goal_point)

    def get_pointing_axis(self):
        return self.get_input_Vector3Stamped(self.pointing_axis)

    def get_constraint(self):
        soft_constraints = OrderedDict()

        weight = self.get_input_float(self.weight)
        root_T_tip = self.get_fk(self.root, self.tip)
        goal_point = self.get_goal_point()
        pointing_axis = self.get_pointing_axis()

        goal_axis = goal_point - sw.position_of(root_T_tip)
        goal_axis /= sw.norm(goal_axis)  # FIXME possible /0
        current_axis = root_T_tip * pointing_axis
        diff = goal_axis - current_axis

        soft_constraints[str(self) + u'x'] = SoftConstraint(lower=diff[0],
                                                            upper=diff[0],
                                                            weight=weight,
                                                            expression=current_axis[0])
        soft_constraints[str(self) + u'y'] = SoftConstraint(lower=diff[1],
                                                            upper=diff[1],
                                                            weight=weight,
                                                            expression=current_axis[1])
        soft_constraints[str(self) + u'z'] = SoftConstraint(lower=diff[2],
                                                            upper=diff[2],
                                                            weight=weight,
                                                            expression=current_axis[2])
        return soft_constraints

    def __str__(self):
        s = super(Pointing, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


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
