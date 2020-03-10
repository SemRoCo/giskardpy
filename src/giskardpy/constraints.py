from __future__ import division

import math
import numbers
from collections import OrderedDict

import numpy as np
from geometry_msgs.msg import Vector3Stamped, Vector3
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message

import giskardpy.identifier as identifier
from giskardpy import symbolic_wrapper as w
from giskardpy.exceptions import GiskardException, ConstraintException
from giskardpy.input_system import PoseStampedInput, Point3Input, Vector3Input, Vector3StampedInput, FrameInput, \
    PointStampedInput, TranslationInput
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.tfwrapper import transform_pose, transform_vector

WEIGHTS = [0] + [6 ** x for x in range(7)]

WEIGHT_MAX = WEIGHTS[-1]
WEIGHT_MIN = WEIGHTS[0]

WEIGHT_COLLISION_AVOIDANCE = WEIGHTS[3]


class Constraint(object):
    def __init__(self, god_map, **kwargs):
        self.god_map = god_map

    def save_params_on_god_map(self, params):
        constraints = self.get_god_map().safe_get_data(identifier.constraints_identifier)
        constraints[str(self)] = params
        self.get_god_map().safe_set_data(identifier.constraints_identifier, constraints)

    def make_constraints(self):
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
        return self.god_map.to_symbol(identifier.sample_period)

    def make_polynomial_function(self, x, p1x, p1y,
                                 p2x, p2y,
                                 min_x, min_y):
        C = min_y
        B = min_x

        order = math.log(((p2y - min_y) / (p1y - min_y)), ((min_x - p2x) / (min_x - p1x)))
        A = (p1y - C) / ((B - p1x) ** order)

        return A * ((-x) + B) ** order + C

    def make_polynomial_function2(self, x, local_max_x, local_max_y,
                                  local_min_x, local_min_y,
                                  order):
        """
        function of form x**order - x**(order-1)
        :return:
        """
        order_1 = order
        order_2 = order - 1
        A = (order_2 / order_1) * (1 / (local_min_x - local_max_x))
        B = (order_1 ** order_1 / order_2 ** order_2) * (local_max_y - local_min_y)
        C = -local_max_x
        D = local_max_y
        return B * ((x + C) * A) ** order_1 - B * ((x + C) * A) ** order_2 + D

    def magic_weight_function(self, x, p1x, p1y,
                              p2x, p2y,
                              saddlex, saddley,
                              min_x, min_y):
        f0 = p1y
        f1 = self.make_polynomial_function(x, p1x, p1y, p2x, p2y, saddlex, saddley)
        f2 = self.make_polynomial_function2(x, saddlex, saddley, min_x, min_y, 3)
        f3 = min_y
        return w.if_less_eq(x, p1x, f0, w.if_less_eq(x, saddlex, f1, w.if_less_eq(x, min_x, f2, f3)))

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

    def limit_acceleration(self, current_position, error, max_acceleration, max_velocity):
        sample_period = self.get_input_sampling_period()
        position_jacobian = w.jacobian(current_position, self.get_robot().get_joint_position_symbols())
        last_velocities = w.Matrix(self.get_robot().get_joint_velocity_symbols())
        last_velocity = w.dot(position_jacobian, last_velocities)[0]
        max_velocity = max_velocity * sample_period
        max_acceleration = max_acceleration * sample_period
        last_velocity = last_velocity * sample_period
        m = 1 / max_acceleration
        error *= m
        max_acceleration *= m
        last_velocity *= m
        max_velocity *= m
        sign = w.sign(error)
        error = w.Abs(error)
        error_rounded = np.floor(error)
        cmd = w.if_greater(max_acceleration, error,
                           error,
                           w.sqrt(error_rounded * 2 * max_acceleration + (
                                   max_acceleration ** 2 / 4)) - max_acceleration / 2)
        cmd *= sign

        vel = w.Max(w.Min(cmd, w.Min(last_velocity + max_acceleration, max_velocity)),
                    w.Max(last_velocity - max_acceleration, -max_velocity))
        return vel / m

    def limit_velocity(self, error, max_velocity):
        sample_period = self.get_input_sampling_period()
        max_velocity *= sample_period
        return w.diffable_max_fast(w.diffable_min_fast(error, max_velocity), -max_velocity)

    def get_constraints(self):
        """
        :rtype: OrderedDict
        """
        self.soft_constraints = OrderedDict()
        self.make_constraints()
        return self.soft_constraints

    def add_constraint(self, name, lower, upper, weight, expression):
        """
        :type name: str
        :type constraint: SoftConstraint
        """
        if name in self.soft_constraints:
            raise KeyError(u'a constraint with name \'{}\' already exists'.format(name))
        self.soft_constraints[name] = SoftConstraint(lower=lower,
                                                     upper=upper,
                                                     weight=weight,
                                                     expression=expression)

    def add_debug_constraint(self, name, expr):
        """
        Adds a constraint with weight 0 to the qp problem.
        Used to inspect subexpressions for debugging.
        :param name: a name to identify the expression
        :type name: str
        :type expr: w.Symbol
        """
        self.add_constraint(name, expr, expr, 1, 0)


class JointPositionContinuous(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, joint_name, goal, weight=WEIGHTS[5], max_velocity=1, max_acceleration=1):
        super(JointPositionContinuous, self).__init__(god_map)
        self.joint_name = joint_name

        if not self.get_robot().is_joint_continuous(joint_name):
            raise ConstraintException(u'{} called with non continuous joint {}'.format(self.__name__,
                                                                                       joint_name))

        params = {self.goal: goal,
                  self.weight: weight,
                  self.max_velocity: max_velocity,
                  self.max_acceleration: max_acceleration}
        self.save_params_on_god_map(params)

    def make_constraints(self):
        """
        example:
        name='JointPosition'
        parameter_value_pair='{
            "joint_name": "torso_lift_joint", #required
            "goal_position": 0, #required
            "weight": 1, #optional
            "max_velocity": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_input_joint_position(self.joint_name)

        joint_goal = self.get_input_float(self.goal)
        weight = self.get_input_float(self.weight)

        max_acceleration = self.get_input_float(self.max_acceleration)
        max_velocity = w.Min(self.get_input_float(self.max_velocity),
                             self.get_robot().get_joint_velocity_limit_expr(self.joint_name))

        error = w.shortest_angular_distance(current_joint, joint_goal)
        # capped_err = self.limit_acceleration(current_joint, error, max_acceleration, max_velocity)
        capped_err = self.limit_velocity(error, max_velocity)

        weight = self.magic_weight_function(w.Abs(error),
                                            0.0, WEIGHTS[5],
                                            np.pi / 8, WEIGHTS[4],
                                            np.pi / 6, WEIGHTS[3],
                                            np.pi / 4, WEIGHTS[1])

        self.add_constraint(str(self), lower=capped_err, upper=capped_err, weight=weight, expression=current_joint)

    def __str__(self):
        s = super(JointPositionContinuous, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionPrismatic(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, joint_name, goal, weight=None, max_velocity=0.1, max_acceleration=0.1):
        super(JointPositionPrismatic, self).__init__(god_map)
        self.joint_name = joint_name
        if not self.get_robot().is_joint_prismatic(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__name__,
                                                                                      joint_name))

        params = {self.goal: goal,
                  self.weight: weight,
                  self.max_velocity: max_velocity,
                  self.max_acceleration: max_acceleration}
        self.save_params_on_god_map(params)

    def make_constraints(self):
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

        max_velocity = w.Min(self.get_input_float(self.max_velocity),
                             self.get_robot().get_joint_velocity_limit_expr(self.joint_name))
        max_acceleration = self.get_input_float(self.max_acceleration)

        err = joint_goal - current_joint
        weight = self.magic_weight_function(w.Abs(err),
                                            0.0, WEIGHTS[5],
                                            0.01, WEIGHTS[4],
                                            0.05, WEIGHTS[3],
                                            0.06, WEIGHTS[1])
        capped_err = self.limit_acceleration(current_joint, err, max_acceleration, max_velocity)

        self.add_constraint(str(self), lower=capped_err, upper=capped_err, weight=weight, expression=current_joint)

    def __str__(self):
        s = super(JointPositionPrismatic, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionRevolute(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, joint_name, goal, weight=None, max_velocity=1, max_acceleration=1):
        super(JointPositionRevolute, self).__init__(god_map)
        self.joint_name = joint_name
        if not self.get_robot().is_joint_revolute(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__name__,
                                                                                      joint_name))

        params = {self.goal: goal,
                  self.weight: weight,
                  self.max_velocity: max_velocity,
                  self.max_acceleration: max_acceleration}
        self.save_params_on_god_map(params)

    def make_constraints(self):
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

        max_velocity = w.Min(self.get_input_float(self.max_velocity),
                             self.get_robot().get_joint_velocity_limit_expr(self.joint_name))

        max_acceleration = self.get_input_float(self.max_acceleration)

        err = joint_goal - current_joint
        capped_err = self.limit_acceleration(current_joint, err, max_acceleration, max_velocity)

        # weight = self.magic_weight_function(w.Abs(err),
        #                                     0.0, WEIGHTS[5],
        #                                     np.pi / 8, WEIGHTS[4],
        #                                     np.pi / 6, WEIGHTS[3],
        #                                     np.pi / 4, WEIGHTS[1])
        weight = WEIGHTS[5]

        self.add_constraint(str(self), lower=capped_err, upper=capped_err, weight=weight, expression=current_joint)

    def __str__(self):
        s = super(JointPositionRevolute, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionList(Constraint):
    def __init__(self, god_map, goal_state, weight=None, max_velocity=None):
        super(JointPositionList, self).__init__(god_map)
        self.constraints = []
        for i, joint_name in enumerate(goal_state[u'name']):
            goal_position = goal_state[u'position'][i]
            params = {u'joint_name': joint_name,
                      u'goal': goal_position}
            if weight is not None:
                params[u'weight'] = weight
            if max_velocity is not None:
                params[u'max_velocity'] = max_velocity
            if self.get_robot().is_joint_continuous(joint_name):
                self.constraints.append(JointPositionContinuous(god_map, **params))
            elif self.get_robot().is_joint_revolute(joint_name):
                self.constraints.append(JointPositionRevolute(god_map, **params))
            elif self.get_robot().is_joint_prismatic(joint_name):
                self.constraints.append(JointPositionPrismatic(god_map, **params))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())


class BasicCartesianConstraint(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.1, max_acceleration=0.1):
        super(BasicCartesianConstraint, self).__init__(god_map)
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
                  self.max_acceleration: max_acceleration,
                  self.max_velocity: max_velocity}
        self.save_params_on_god_map(params)

    def get_goal_pose(self):
        return self.get_input_PoseStamped(self.goal)

    def __str__(self):
        s = super(BasicCartesianConstraint, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class CartesianPosition(BasicCartesianConstraint):

    def make_constraints(self):
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
            "max_velocity": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """

        r_P_g = w.position_of(self.get_goal_pose())
        max_velocity = self.get_input_float(self.max_velocity)
        max_acceleration = self.get_input_float(self.max_acceleration)

        r_P_c = w.position_of(self.get_fk(self.root, self.tip))

        r_P_error = r_P_g - r_P_c
        trans_error = w.norm(r_P_error)

        trans_scale = self.limit_acceleration(w.norm(r_P_c),
                                              trans_error,
                                              max_acceleration,
                                              max_velocity)
        r_P_intermediate_error = w.save_division(r_P_error, trans_error) * trans_scale

        weight = self.magic_weight_function(trans_error,
                                            0.0, WEIGHTS[5],
                                            0.01, WEIGHTS[4],
                                            0.05, WEIGHTS[3],
                                            0.06, WEIGHTS[1])

        self.add_constraint(str(self) + u'x', lower=r_P_intermediate_error[0],
                            upper=r_P_intermediate_error[0],
                            weight=weight,
                            expression=r_P_c[0])
        self.add_constraint(str(self) + u'y', lower=r_P_intermediate_error[1],
                            upper=r_P_intermediate_error[1],
                            weight=weight,
                            expression=r_P_c[1])
        self.add_constraint(str(self) + u'z', lower=r_P_intermediate_error[2],
                            upper=r_P_intermediate_error[2],
                            weight=weight,
                            expression=r_P_c[2])


# class CartesianPositionX(BasicCartesianConstraint):
# FIXME
#     def make_constraints(self):
#         goal_position = w.position_of(self.get_goal_pose())
#         weight = self.get_input_float(self.weight)
#         max_velocity = self.get_input_float(self.max_velocity)
#         t = self.get_input_sampling_period()
#
#         current_position = w.position_of(self.get_fk(self.root, self.tip))
#
#         trans_error_vector = goal_position - current_position
#         trans_error = w.norm(trans_error_vector)
#         trans_scale = w.diffable_min_fast(trans_error, max_velocity * t)
#         trans_control = w.save_division(trans_error_vector, trans_error) * trans_scale
#
#         self.add_constraint(str(self) + u'x', lower=trans_control[0],
#                             upper=trans_control[0],
#                             weight=weight,
#                             expression=current_position[0])
#
#
# class CartesianPositionY(BasicCartesianConstraint):
#     def make_constraints(self):
#         goal_position = w.position_of(self.get_goal_pose())
#         weight = self.get_input_float(self.weight)
#         max_velocity = self.get_input_float(self.max_velocity)
#         t = self.get_input_sampling_period()
#
#         current_position = w.position_of(self.get_fk(self.root, self.tip))
#
#         trans_error_vector = goal_position - current_position
#         trans_error = w.norm(trans_error_vector)
#         trans_scale = w.diffable_min_fast(trans_error, max_velocity * t)
#         trans_control = w.save_division(trans_error_vector, trans_error) * trans_scale
#
#         self.add_constraint(str(self) + u'y', lower=trans_control[1],
#                             upper=trans_control[1],
#                             weight=weight,
#                             expression=current_position[1])


class CartesianOrientation(BasicCartesianConstraint):
    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.5, max_acceleration=0.5):
        super(CartesianOrientation, self).__init__(god_map, root_link, tip_link, goal, max_velocity, max_acceleration)

    def make_constraints(self):
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
            "max_velocity": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        goal_rotation = w.rotation_of(self.get_goal_pose())
        max_velocity = self.get_input_float(self.max_velocity)
        max_acceleration = self.get_input_float(self.max_acceleration)

        current_rotation = w.rotation_of(self.get_fk(self.root, self.tip))
        current_evaluated_rotation = w.rotation_of(self.get_fk_evaluated(self.root, self.tip))

        hack = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

        axis, current_angle = w.diffable_axis_angle_from_matrix(
            w.dot(w.dot(current_evaluated_rotation.T, hack), current_rotation))
        c_aa = (axis * current_angle)

        axis, angle = w.diffable_axis_angle_from_matrix(w.dot(current_rotation.T, goal_rotation))

        capped_angle = self.limit_acceleration(current_angle,
                                               angle,
                                               max_acceleration,
                                               max_velocity)

        r_rot_control = axis * capped_angle

        weight = WEIGHTS[5]

        self.add_constraint(str(self) + u'/0', lower=r_rot_control[0],
                            upper=r_rot_control[0],
                            weight=weight,
                            expression=c_aa[0])
        self.add_constraint(str(self) + u'/1', lower=r_rot_control[1],
                            upper=r_rot_control[1],
                            weight=weight,
                            expression=c_aa[1])
        self.add_constraint(str(self) + u'/2', lower=r_rot_control[2],
                            upper=r_rot_control[2],
                            weight=weight,
                            expression=c_aa[2])


class CartesianOrientationSlerp(BasicCartesianConstraint):
    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.5, max_accleration=0.5):
        super(CartesianOrientationSlerp, self).__init__(god_map, root_link, tip_link, goal, max_velocity,
                                                        max_accleration)

    def make_constraints(self):
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
            "max_velocity": 0.3 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        r_R_g = w.rotation_of(self.get_goal_pose())
        weight = self.get_input_float(self.weight)
        max_velocity = self.get_input_float(self.max_velocity)
        max_acceleration = self.get_input_float(self.max_acceleration)

        r_R_c = w.rotation_of(self.get_fk(self.root, self.tip))
        r_R_c_evaluated = w.rotation_of(self.get_fk_evaluated(self.root, self.tip))

        identity = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
        c_R_c = w.dot(w.dot(r_R_c_evaluated.T, identity), r_R_c)
        current_axis, current_angle = w.diffable_axis_angle_from_matrix(c_R_c)
        current_angle_axis = (current_axis * current_angle)

        error_angle = w.rotation_distance(r_R_c, r_R_g)
        error_angle = w.diffable_abs(error_angle)

        _, angle = w.diffable_axis_angle_from_matrix(r_R_c)
        # capped_angle = self.limit_acceleration(angle,
        #                                        error_angle,
        #                                        max_acceleration,
        #                                        max_velocity) / error_angle

        capped_angle = self.limit_velocity(error_angle, max_velocity) / error_angle

        r_R_c_q = w.quaternion_from_matrix(r_R_c)
        r_R_g_q = w.quaternion_from_matrix(r_R_g)
        r_R_g_intermediate_q = w.diffable_slerp(r_R_c_q, r_R_g_q, capped_angle)
        c_R_g_intermediate_q = w.quaternion_diff(r_R_c_q, r_R_g_intermediate_q)
        intermediate_error_axis, intermediate_error_angle = w.axis_angle_from_quaternion(c_R_g_intermediate_q[0],
                                                                                         c_R_g_intermediate_q[1],
                                                                                         c_R_g_intermediate_q[2],
                                                                                         c_R_g_intermediate_q[3])

        c_R_g_intermediate_aa = intermediate_error_axis * intermediate_error_angle

        weight = WEIGHTS[3]

        self.add_constraint(str(self) + u'/0', lower=c_R_g_intermediate_aa[0],
                            upper=c_R_g_intermediate_aa[0],
                            weight=weight,
                            expression=current_angle_axis[0])
        self.add_constraint(str(self) + u'/1', lower=c_R_g_intermediate_aa[1],
                            upper=c_R_g_intermediate_aa[1],
                            weight=weight,
                            expression=current_angle_axis[1])
        self.add_constraint(str(self) + u'/2', lower=c_R_g_intermediate_aa[2],
                            upper=c_R_g_intermediate_aa[2],
                            weight=weight,
                            expression=current_angle_axis[2])


class CartesianPose(Constraint):
    def __init__(self, god_map, root_link, tip_link, goal, translation_max_velocity=0.1,
                 translation_max_acceleration=0.1, rotation_max_velocity=0.5, rotation_max_acceleration=0.5):
        super(CartesianPose, self).__init__(god_map)
        self.constraints = []
        self.constraints.append(CartesianPosition(god_map, root_link, tip_link, goal,
                                                  translation_max_velocity, translation_max_acceleration))
        self.constraints.append(CartesianOrientationSlerp(god_map, root_link, tip_link, goal,
                                                          rotation_max_velocity, rotation_max_acceleration))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.make_constraints())


class ExternalCollisionAvoidance(Constraint):
    repel_velocity = u'repel_velocity'
    max_weight_distance = u'max_weight_distance'
    low_weight_distance = u'low_weight_distance'
    zero_weight_distance = u'zero_weight_distance'
    root_T_link_b = u'root_T_link_b'
    link_in_chain = u'link_in_chain'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, link_name, repel_velocity=0.1, max_weight_distance=0.0, low_weight_distance=0.01,
                 zero_weight_distance=0.05, idx=0, max_acceleration=0.005):
        super(ExternalCollisionAvoidance, self).__init__(god_map)
        self.link_name = link_name
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()
        self.idx = idx

        params = {self.repel_velocity: repel_velocity,
                  self.max_weight_distance: max_weight_distance,
                  self.low_weight_distance: low_weight_distance,
                  self.zero_weight_distance: zero_weight_distance,
                  self.max_acceleration: max_acceleration}
        self.save_params_on_god_map(params)

    def get_contact_normal_on_b_in_root(self):
        return Vector3Input(self.god_map.to_symbol,
                            prefix=identifier.closest_point + [u'get_external_collisions',
                                                               (self.link_name,),
                                                               self.idx,
                                                               u'get_contact_normal_in_root',
                                                               tuple()]).get_expression()

    def get_closest_point_on_a_in_a(self):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [u'get_external_collisions',
                                                              (self.link_name,),
                                                              self.idx,
                                                              u'get_position_on_a_in_a',
                                                              tuple()]).get_expression()

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions',
                                                                  (self.link_name,),
                                                                  self.idx,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def make_constraints(self):
        a_P_pa = self.get_closest_point_on_a_in_a()
        r_V_n = self.get_contact_normal_on_b_in_root()
        actual_distance = self.get_actual_distance()
        repel_velocity = self.get_input_float(self.repel_velocity)
        max_acceleration = self.get_input_float(self.max_acceleration)
        zero_weight_distance = self.get_input_float(self.zero_weight_distance)

        r_T_a = self.get_fk(self.robot_root, self.link_name)

        r_P_pa = w.dot(r_T_a, a_P_pa)

        dist = w.dot(r_V_n.T, r_P_pa)[0]

        weight_f = self.magic_weight_function(actual_distance,
                                              0.0, WEIGHT_MAX,  #nothing should be stronger than ca
                                              0.01, WEIGHTS[4], # everything should stay below this to ensure ca is strong enough
                                              0.05, WEIGHTS[2], #ca active but can get overpowered
                                              0.06, WEIGHT_MIN) #everything is stronger than ca

        penetration_distance = zero_weight_distance - actual_distance

        limit = self.limit_acceleration(dist,
                                        penetration_distance,
                                        max_acceleration,
                                        repel_velocity)

        # limit = self.limit_velocity(actual_distance, repel_velocity)

        self.add_constraint(str(self), lower=limit,
                            upper=1e9,
                            weight=weight_f,
                            expression=dist)

    def __str__(self):
        s = super(ExternalCollisionAvoidance, self).__str__()
        return u'{}/{}/{}'.format(s, self.link_name, self.idx)


class SelfCollisionAvoidance(Constraint):
    repel_velocity = u'repel_velocity'
    max_weight_distance = u'max_weight_distance'
    low_weight_distance = u'low_weight_distance'
    zero_weight_distance = u'zero_weight_distance'
    root_T_link_b = u'root_T_link_b'
    link_in_chain = u'link_in_chain'

    def __init__(self, god_map, link_a, link_b, repel_velocity=0.1, max_weight_distance=0.0, low_weight_distance=0.01,
                 zero_weight_distance=0.05, idx=0):
        super(SelfCollisionAvoidance, self).__init__(god_map)
        self.link_a = link_a
        self.link_b = link_b
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()
        self.idx = idx

        params = {self.repel_velocity: repel_velocity,
                  self.max_weight_distance: max_weight_distance,
                  self.low_weight_distance: low_weight_distance,
                  self.zero_weight_distance: zero_weight_distance, }
        self.save_params_on_god_map(params)

    def get_contact_normal_in_b(self):
        return Vector3Input(self.god_map.to_symbol,
                            prefix=identifier.closest_point + [u'get_self_collisions',
                                                               (self.link_a, self.link_b),
                                                               self.idx,
                                                               u'get_contact_normal_in_b',
                                                               tuple()]).get_expression()

    def get_position_on_a_in_a(self):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [u'get_self_collisions',
                                                              (self.link_a, self.link_b),
                                                              self.idx,
                                                              u'get_position_on_a_in_a',
                                                              tuple()]).get_expression()

    def get_b_T_pb(self):
        return TranslationInput(self.god_map.to_symbol,
                                prefix=identifier.closest_point + [u'get_self_collisions',
                                                                   (self.link_a, self.link_b),
                                                                   self.idx,
                                                                   u'get_position_on_b_in_b',
                                                                   tuple()]).get_frame()

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_self_collisions',
                                                                  (self.link_a, self.link_b),
                                                                  self.idx,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def make_constraints(self):
        repel_velocity = self.get_input_float(self.repel_velocity)
        zero_weight_distance = self.get_input_float(self.zero_weight_distance)
        actual_distance = self.get_actual_distance()

        movable_joint = self.get_robot().get_controlled_parent_joint(self.link_a)
        f = self.get_robot().get_child_link_of_joint(movable_joint)
        a_T_f = self.get_fk_evaluated(self.link_a, f)

        b_T_a = self.get_fk(self.link_b, self.link_a)
        pb_T_b = w.inverse_frame(self.get_b_T_pb())
        f_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        a_P_pa = w.dot(a_T_f, f_P_pa)

        pb_P_pa = w.dot(pb_T_b, b_T_a, a_P_pa)

        dist = w.dot(pb_V_n.T, pb_P_pa)[0]

        weight_f = self.magic_weight_function(actual_distance,
                                              0.0, WEIGHT_MAX,
                                              0.01, WEIGHTS[4],
                                              0.05, WEIGHTS[2],
                                              0.06, WEIGHT_MIN)

        limit = zero_weight_distance - actual_distance
        limit = self.limit_velocity(limit, repel_velocity)

        self.add_constraint(str(self), lower=limit, upper=1e9, weight=weight_f, expression=dist)

    def __str__(self):
        s = super(SelfCollisionAvoidance, self).__str__()
        return u'{}/{}/{}/{}'.format(s, self.link_a, self.link_b, self.idx)


class AlignPlanes(Constraint):
    root_normal = u'root_normal'
    tip_normal = u'tip_normal'
    max_velocity = u'max_velocity'

    def __init__(self, god_map, root, tip, root_normal, tip_normal, max_velocity=0.5):
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
                  self.max_velocity: max_velocity}
        self.save_params_on_god_map(params)

    def __str__(self):
        s = super(AlignPlanes, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)

    def get_root_normal_vector(self):
        return self.get_input_Vector3Stamped(self.root_normal)

    def get_tip_normal_vector(self):
        return self.get_input_Vector3Stamped(self.tip_normal)

    def make_constraints(self):
        # TODO integrate max_velocity?
        weight = self.get_input_float(self.weight)
        # max_velocity = self.get_input_float(self.max_velocity)
        root_R_tip = w.rotation_of(self.get_fk(self.root, self.tip))
        tip_normal__tip = self.get_tip_normal_vector()
        root_normal__root = self.get_root_normal_vector()

        tip_normal__root = w.dot(root_R_tip, tip_normal__tip)
        diff = root_normal__root - tip_normal__root

        weight = WEIGHTS[4]

        self.add_constraint(str(self) + u'x', lower=diff[0],
                            upper=diff[0],
                            weight=weight,
                            expression=tip_normal__root[0])
        self.add_constraint(str(self) + u'y', lower=diff[1],
                            upper=diff[1],
                            weight=weight,
                            expression=tip_normal__root[1])
        self.add_constraint(str(self) + u'z', lower=diff[2],
                            upper=diff[2],
                            weight=weight,
                            expression=tip_normal__root[2])


class GravityJoint(Constraint):
    weight = u'weight'

    def __init__(self, god_map, joint_name, object_name):
        super(GravityJoint, self).__init__(god_map)
        self.joint_name = joint_name
        self.object_name = object_name
        params = {}
        self.save_params_on_god_map(params)

    def make_constraints(self):
        current_joint = self.get_input_joint_position(self.joint_name)

        parent_link = self.get_robot().get_parent_link_of_joint(self.joint_name)

        parent_R_root = w.rotation_of(self.get_fk(parent_link, self.get_robot().get_root()))

        com__parent = w.position_of(self.get_fk_evaluated(parent_link, self.object_name))
        com__parent[3] = 0
        com__parent = w.scale(com__parent, 1)

        g = w.vector3(0, 0, -1)
        g__parent = w.dot(parent_R_root, g)
        axis_of_rotation = w.vector3(*self.get_robot().get_joint_axis(self.joint_name))
        l = w.dot(g__parent, axis_of_rotation)
        goal__parent = g__parent - w.scale(axis_of_rotation, l)
        goal__parent = w.scale(goal__parent, 1)

        goal_vel = w.acos(w.dot(com__parent, goal__parent))

        ref_axis_of_rotation = w.cross(com__parent, goal__parent)
        goal_vel *= w.sign(w.dot(ref_axis_of_rotation, axis_of_rotation))

        weight = WEIGHTS[3]

        self.add_constraint(str(self), lower=goal_vel,  # sw.Min(goal_vel, 0),
                            upper=goal_vel,  # sw.Max(goal_vel, 0),
                            weight=weight,
                            expression=current_joint)

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

# FIXME
# class Pointing(Constraint):
#     goal_point = u'goal_point'
#     pointing_axis = u'pointing_axis'
#     weight = u'weight'
#
#     def __init__(self, god_map, tip, goal_point, root=None, pointing_axis=None, weight=HIGH_WEIGHT):
#         """
#         :type tip: str
#         :param goal_point: json representing PointStamped
#         :type goal_point: str
#         :param root: default is root link of robot
#         :type root: str
#         :param pointing_axis: json representing Vector3Stamped, default is z axis
#         :type pointing_axis: str
#         :type weight: float
#         """
#         super(Pointing, self).__init__(god_map)
#         if root is None:
#             self.root = self.get_robot().get_root()
#         else:
#             self.root = root
#         self.tip = tip
#
#         goal_point = convert_dictionary_to_ros_message(u'geometry_msgs/PointStamped', goal_point)
#         goal_point = transform_point(self.root, goal_point)
#
#         if pointing_axis is not None:
#             pointing_axis = convert_dictionary_to_ros_message(u'geometry_msgs/Vector3Stamped', pointing_axis)
#             pointing_axis = transform_vector(self.tip, pointing_axis)
#         else:
#             pointing_axis = Vector3Stamped()
#             pointing_axis.header.frame_id = self.tip
#             pointing_axis.vector.z = 1
#         tmp = np.array([pointing_axis.vector.x, pointing_axis.vector.y, pointing_axis.vector.z])
#         tmp = tmp / np.linalg.norm(tmp)  # TODO possible /0
#         pointing_axis.vector = Vector3(*tmp)
#
#         params = {self.goal_point: goal_point,
#                   self.pointing_axis: pointing_axis,
#                   self.weight: weight}
#         self.save_params_on_god_map(params)
#
#     def get_goal_point(self):
#         return self.get_input_PointStamped(self.goal_point)
#
#     def get_pointing_axis(self):
#         return self.get_input_Vector3Stamped(self.pointing_axis)
#
#     def make_constraints(self):
#
#         weight = self.get_input_float(self.weight)
#         root_T_tip = self.get_fk(self.root, self.tip)
#         goal_point = self.get_goal_point()
#         pointing_axis = self.get_pointing_axis()
#
#         goal_axis = goal_point - w.position_of(root_T_tip)
#         goal_axis /= w.norm(goal_axis)  # FIXME possible /0
#         current_axis = w.dot(root_T_tip, pointing_axis)
#         diff = goal_axis - current_axis
#
#         self.add_constraint(str(self) + u'x', lower=diff[0],
#                             upper=diff[0],
#                             weight=weight,
#                             expression=current_axis[0])
#         self.add_constraint(str(self) + u'y', lower=diff[1],
#                             upper=diff[1],
#                             weight=weight,
#                             expression=current_axis[1])
#         self.add_constraint(str(self) + u'z', lower=diff[2],
#                             upper=diff[2],
#                             weight=weight,
#                             expression=current_axis[2])
#
#     def __str__(self):
#         s = super(Pointing, self).__str__()
#         return u'{}/{}/{}'.format(s, self.root, self.tip)
