from __future__ import division

import numbers
import random
import time
from collections import OrderedDict
from copy import deepcopy

import PyKDL as kdl
import casadi
import numpy as np
from scipy import interpolate
import rospy
from geometry_msgs.msg import Vector3Stamped, Vector3
from giskard_msgs.msg import Constraint as Constraint_msg

import giskardpy.identifier as identifier
import giskardpy.tfwrapper as tf
from giskardpy import casadi_wrapper as w
from giskardpy.data_types import SoftConstraint
from giskardpy.exceptions import GiskardException, ConstraintException
from giskardpy.input_system import \
    PoseStampedInput, Point3Input, Vector3Input, \
    Vector3StampedInput, FrameInput, \
    PointStampedInput, TranslationInput
from giskardpy.logging import logwarn
from rospy_message_converter.message_converter import \
    convert_dictionary_to_ros_message, \
    convert_ros_message_to_dictionary

WEIGHT_MAX = Constraint_msg.WEIGHT_MAX # 1000
WEIGHT_ABOVE_CA = Constraint_msg.WEIGHT_ABOVE_CA #100
WEIGHT_COLLISION_AVOIDANCE = Constraint_msg.WEIGHT_COLLISION_AVOIDANCE #10
WEIGHT_BELOW_CA = Constraint_msg.WEIGHT_BELOW_CA # 1
WEIGHT_MIN = Constraint_msg.WEIGHT_MIN # 0


class Constraint(object):
    def __init__(self, god_map, **kwargs):
        self.god_map = god_map

    def save_params_on_god_map(self, params):
        constraints = self.get_god_map().get_data(identifier.constraints_identifier)
        try:
            constraints[str(self)].update(params)
        except:
            constraints[str(self)] = params

        self.get_god_map().set_data(identifier.constraints_identifier, constraints)

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
        return self.get_god_map().get_data(identifier.world)

    def get_robot(self):
        """
        :rtype: giskardpy.robot.Robot
        """
        return self.get_god_map().get_data(identifier.robot)

    def get_world_unsafe(self):
        """
        :rtype: giskardpy.world.World
        """
        return self.get_god_map().unsafe_get_data(identifier.world)

    def get_robot_unsafe(self):
        """
        :rtype: giskardpy.robot.Robot
        """
        return self.get_god_map().unsafe_get_data(identifier.robot)

    def get_input_joint_position(self, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        key = identifier.joint_states + [joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_input_object_joint_position(self, object_name, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        key = identifier.world + [u'get_object', (object_name,), u'joint_state', joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_input_sampling_period(self):
        return self.god_map.to_symbol(identifier.sample_period)

    # def make_polynomial_function(self, x, p1x, p1y,
    #                              p2x, p2y,
    #                              min_x, min_y):
    #     C = min_y
    #     B = min_x
    #
    #     order = math.log(((p2y - min_y) / (p1y - min_y)), ((min_x - p2x) / (min_x - p1x)))
    #     A = (p1y - C) / ((B - p1x) ** order)
    #
    #     return A * ((-x) + B) ** order + C
    #
    # def make_polynomial_function2(self, x, local_max_x, local_max_y,
    #                               local_min_x, local_min_y,
    #                               order):
    #     """
    #     function of form x**order - x**(order-1)
    #     :return:
    #     """
    #     order_1 = order
    #     order_2 = order - 1
    #     A = (order_2 / order_1) * (1 / (local_min_x - local_max_x))
    #     B = (order_1 ** order_1 / order_2 ** order_2) * (local_max_y - local_min_y)
    #     C = -local_max_x
    #     D = local_max_y
    #     return B * ((x + C) * A) ** order_1 - B * ((x + C) * A) ** order_2 + D
    #
    # def magic_weight_function(self, x, p1x, p1y,
    #                           p2x, p2y,
    #                           saddlex, saddley,
    #                           min_x, min_y):
    #     f0 = p1y
    #     f1 = self.make_polynomial_function(x, p1x, p1y, p2x, p2y, saddlex, saddley)
    #     f2 = self.make_polynomial_function2(x, saddlex, saddley, min_x, min_y, 3)
    #     f3 = min_y
    #     return w.if_less_eq(x, p1x, f0, w.if_less_eq(x, saddlex, f1, w.if_less_eq(x, min_x, f2, f3)))

    def __str__(self):
        return self.__class__.__name__

    def get_fk(self, root, tip):
        """
        Return the homogeneous transformation matrix root_T_tip as a function that is dependent on the joint state.
        :type root: str
        :type tip: str
        :return: root_T_tip
        """
        return self.get_robot().get_fk_expression(root, tip)

    def get_fk_evaluated(self, root, tip):
        """
        Return the homogeneous transformation matrix root_T_tip. This Matrix refers to the evaluated current transform.
        It is not dependent on the joint state.
        :type root: str
        :type tip: str
        :return: root_T_tip
        """
        return FrameInput(self.get_god_map().to_symbol,
                          prefix=identifier.fk_np +
                                 [(root, tip)]).get_frame()

    def get_input_float(self, name):
        """
        Returns a symbol that refers to the value of "name" on god map
        :type name: Union[str, unicode]
        :return: symbol
        """
        key = self.get_identifier() + [name]
        return self.god_map.to_symbol(key)

    def parse_and_transform_PoseStamped(self, pose_stamped_json, goal_reference_frame):
        """
        Takes a pose stamped json, turns it into a ros message and transforms it into the goal frame
        :param pose_stamped_json: json representing a pose stamped
        :type pose_stamped_json: str
        :param goal_reference_frame: name of the goal frame
        :type goal_reference_frame: str
        :return:
        """
        result = convert_dictionary_to_ros_message(u'geometry_msgs/PoseStamped', pose_stamped_json)
        result = tf.transform_pose(goal_reference_frame, result)
        result.pose.orientation = tf.normalize(result.pose.orientation)
        return result

    def parse_and_transform_Vector3Stamped(self, vector3_stamped_json, goal_reference_frame, normalized=False):
        result = convert_dictionary_to_ros_message(u'geometry_msgs/Vector3Stamped', vector3_stamped_json)
        result = tf.transform_vector(goal_reference_frame, result)
        if normalized:
            result.vector = tf.normalize(result.vector)
        return result

    def parse_and_transform_PointStamped(self, point_stamped_json, goal_reference_frame):
        result = convert_dictionary_to_ros_message(u'geometry_msgs/PointStamped', point_stamped_json)
        result = tf.transform_point(goal_reference_frame, result)
        return result

    def get_input_PoseStamped(self, name):
        """
        :param name: name of the god map entry
        :return: a homogeneous transformation matrix, with symbols that refer to a pose stamped in the god map.
        """
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

    def get_input_np_frame(self, name):
        return FrameInput(self.get_god_map().to_symbol,
                          prefix=self.get_identifier() + [name]).get_frame()

    def get_expr_velocity(self, expr):
        expr_jacobian = w.jacobian(expr, self.get_robot().get_joint_position_symbols())
        last_velocities = w.Matrix(self.get_robot().get_joint_velocity_symbols())
        velocity = w.dot(expr_jacobian, last_velocities)
        if velocity.shape[0] * velocity.shape[0] == 1:
            return velocity[0]
        else:
            return velocity

    def get_fk_velocity(self, root, tip):
        r_T_t = self.get_fk(root, tip)
        r_R_t = w.rotation_of(r_T_t)
        axis, angle = w.axis_angle_from_matrix(r_R_t)
        r_R_t_axis_angle = axis * angle
        r_P_t = w.position_of(r_T_t)
        fk = w.Matrix([r_P_t[0],
                       r_P_t[1],
                       r_P_t[2],
                       r_R_t_axis_angle[0],
                       r_R_t_axis_angle[1],
                       r_R_t_axis_angle[2]])
        return self.get_expr_velocity(fk)

    def limit_acceleration(self, current_position, error, max_acceleration, max_velocity, debug_prefix=None):
        """
        experimental, don't use
        """
        sample_period = self.get_input_sampling_period()
        last_velocity = self.get_expr_velocity(current_position)
        if debug_prefix is not None:
            self.add_debug_constraint(debug_prefix + '/velocity', last_velocity)
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

    def add_max_force_constraint(self, name, expression):
        expr_jacobian = w.jacobian(expression, self.get_robot().get_joint_position_symbols())
        total_derivative = w.Sum(w.Abs(expr_jacobian))
        self.add_constraint(name,
                            lower=100,
                            upper=100,
                            weight=10,
                            expression=total_derivative,
                            goal_constraint=False)

    def limit_velocity(self, error, max_velocity):
        """
        :param error: expression that describes the error
        :param max_velocity: float or expression representing the max velocity
        :return: expression that limits the velocity of error to max_velocity
        """
        sample_period = self.get_input_sampling_period()
        max_velocity *= sample_period
        return w.Max(w.Min(error, max_velocity), -max_velocity)

    def normalize_weight(self, velocity_limit, weight):
        sample_period = self.get_input_sampling_period()
        result = weight * (1. / (sample_period * velocity_limit)) ** 2
        return result

    def get_constraints(self):
        """
        :rtype: OrderedDict
        """
        self.soft_constraints = OrderedDict()
        self.make_constraints()
        return self.soft_constraints

    def add_constraint(self, name_suffix, lower, upper, weight, expression, goal_constraint=False,
                       lower_slack_limit=-1e9,
                       upper_slack_limit=1e9, linear_weight=0):
        """
        :param name_suffix: name of the constraint, make use to avoid name conflicts!
        :type name_suffix: Union[str, unicode]
        :param lower: lower limit for the !derivative! of the expression
        :type lower: float, or symbolic expression
        :param upper: upper limit for the !derivative! of the expression
        :type upper: float, or symbolic expression
        :param weight: tells the solver how important this constraint is, if unsure, use HIGH_WEIGHT
        :param expression: symbolic expression that describes a geometric property. make sure it as a depedency on the
                            joint state. usually achieved through "get_fk"
        """
        name = str(self) + name_suffix
        if name in self.soft_constraints:
            raise KeyError(u'a constraint with name \'{}\' already exists'.format(name))
        self.soft_constraints[name] = SoftConstraint(lbA=lower,
                                                     ubA=upper,
                                                     weight=weight,
                                                     expression=expression,
                                                     goal_constraint=goal_constraint,
                                                     lower_slack_limit=lower_slack_limit,
                                                     upper_slack_limit=upper_slack_limit,
                                                     linear_weight=linear_weight)

    def add_debug_constraint(self, name, expr):
        """
        Adds a constraint with weight 0 to the qp problem.
        Used to inspect subexpressions for debugging.
        :param name: a name to identify the expression
        :type name: str
        :type expr: w.Symbol
        """
        self.add_constraint(name, expr, expr, 1, 0, False)

    def add_debug_matrix(self, name, matrix_expr):
        for x in range(matrix_expr.shape[0]):
            for y in range(matrix_expr.shape[1]):
                self.add_debug_constraint(name + u'/{},{}'.format(x, y), matrix_expr[x, y])

    def add_debug_vector(self, name, vector_expr):
        for x in range(vector_expr.shape[0]):
            self.add_debug_constraint(name + u'/{}'.format(x), vector_expr[x])

    def add_minimize_position_constraints(self, r_P_g, max_velocity, max_acceleration, root, tip, goal_constraint,
                                          weight=WEIGHT_BELOW_CA, prefix=u''):
        """
        :param r_P_g: position of goal relative to root frame
        :param max_velocity:
        :param max_acceleration:
        :param root:
        :param tip:
        :param prefix: name prefix to distinguish different constraints
        :type prefix: str
        :return:
        """
        r_P_c = w.position_of(self.get_fk(root, tip))

        r_P_error = r_P_g - r_P_c
        trans_error = w.norm(r_P_error)

        trans_scale = self.limit_velocity(trans_error, max_velocity)
        r_P_intermediate_error = w.save_division(r_P_error, trans_error) * trans_scale

        # weight = self.magic_weight_function(trans_error,
        #                                     0.0, WEIGHTS[5],
        #                                     0.01, WEIGHTS[4],
        #                                     0.05, WEIGHTS[3],
        #                                     0.06, WEIGHTS[1])
        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'/{}/x'.format(prefix),
                            lower=r_P_intermediate_error[0],
                            upper=r_P_intermediate_error[0],
                            weight=weight,
                            expression=r_P_c[0],
                            goal_constraint=goal_constraint)
        self.add_constraint(u'/{}/y'.format(prefix),
                            lower=r_P_intermediate_error[1],
                            upper=r_P_intermediate_error[1],
                            weight=weight,
                            expression=r_P_c[1],
                            goal_constraint=goal_constraint)
        self.add_constraint(u'/{}/z'.format(prefix),
                            lower=r_P_intermediate_error[2],
                            upper=r_P_intermediate_error[2],
                            weight=weight,
                            expression=r_P_c[2],
                            goal_constraint=goal_constraint)

    def add_minimize_vector_angle_constraints(self, max_velocity, root, tip, tip_V_tip_normal, root_V_goal_normal,
                                              weight=WEIGHT_BELOW_CA, goal_constraint=False, prefix=u''):
        root_R_tip = w.rotation_of(self.get_fk(root, tip))
        root_V_tip_normal = w.dot(root_R_tip, tip_V_tip_normal)

        angle = w.acos(w.dot(root_V_tip_normal.T, root_V_goal_normal)[0])
        angle_limited = self.limit_velocity(angle, max_velocity) / angle
        root_V_goal_normal_intermediate = w.slerp(root_V_tip_normal, root_V_goal_normal, angle_limited)
        error = root_V_goal_normal_intermediate - root_V_tip_normal

        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'/{}/rot/x'.format(prefix),
                            lower=error[0],
                            upper=error[0],
                            weight=weight,
                            expression=root_V_tip_normal[0],
                            goal_constraint=goal_constraint)
        self.add_constraint(u'/{}/rot/y'.format(prefix),
                            lower=error[1],
                            upper=error[1],
                            weight=weight,
                            expression=root_V_tip_normal[1],
                            goal_constraint=goal_constraint)
        self.add_constraint(u'/{}/rot/z'.format(prefix),
                            lower=error[2],
                            upper=error[2],
                            weight=weight,
                            expression=root_V_tip_normal[2],
                            goal_constraint=goal_constraint)

    def add_minimize_rotation_constraints(self, root_R_tipGoal, root, tip, max_velocity=np.pi / 4,
                                          weight=WEIGHT_BELOW_CA, goal_constraint=True, prefix=u''):
        root_R_tipCurrent = w.rotation_of(self.get_fk(root, tip))
        root_R_tipCurrent_evaluated = w.rotation_of(self.get_fk_evaluated(root, tip))

        identity = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
        tipCurrentEvaluated_R_tipCurrent = w.dot(w.dot(root_R_tipCurrent_evaluated.T, identity), root_R_tipCurrent)
        current_axis, current_angle = w.axis_angle_from_matrix(tipCurrentEvaluated_R_tipCurrent)
        current_angle_axis = (current_axis * current_angle)

        error_angle = w.rotation_distance(root_R_tipCurrent, root_R_tipGoal)
        error_angle = w.Abs(error_angle)

        _, angle = w.axis_angle_from_matrix(root_R_tipCurrent)

        capped_angle = self.limit_velocity(error_angle, max_velocity) / error_angle

        r_R_c_q = w.quaternion_from_matrix(root_R_tipCurrent)
        r_R_g_q = w.quaternion_from_matrix(root_R_tipGoal)
        r_R_g_intermediate_q = w.quaternion_slerp(r_R_c_q, r_R_g_q, capped_angle)
        c_R_g_intermediate_q = w.quaternion_diff(r_R_c_q, r_R_g_intermediate_q)
        intermediate_error_axis, intermediate_error_angle = w.axis_angle_from_quaternion(c_R_g_intermediate_q[0],
                                                                                         c_R_g_intermediate_q[1],
                                                                                         c_R_g_intermediate_q[2],
                                                                                         c_R_g_intermediate_q[3])

        c_R_g_intermediate_aa = intermediate_error_axis * intermediate_error_angle

        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'/{}/rot/0'.format(prefix),
                            lower=c_R_g_intermediate_aa[0],
                            upper=c_R_g_intermediate_aa[0],
                            weight=weight,
                            expression=current_angle_axis[0],
                            goal_constraint=goal_constraint)
        self.add_constraint(u'/{}/rot/1'.format(prefix),
                            lower=c_R_g_intermediate_aa[1],
                            upper=c_R_g_intermediate_aa[1],
                            weight=weight,
                            expression=current_angle_axis[1],
                            goal_constraint=goal_constraint)
        self.add_constraint(u'/{}/rot/2'.format(prefix),
                            lower=c_R_g_intermediate_aa[2],
                            upper=c_R_g_intermediate_aa[2],
                            weight=weight,
                            expression=current_angle_axis[2],
                            goal_constraint=goal_constraint)


class JointPositionContinuous(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'
    goal_constraint = u'goal_constraint'

    def __init__(self, god_map, joint_name, goal, weight=WEIGHT_BELOW_CA, max_velocity=1423, max_acceleration=1,
                 goal_constraint=False):
        """
        This goal will move a continuous joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 1423, meaning the urdf/config limits are active
        """
        super(JointPositionContinuous, self).__init__(god_map)
        self.joint_name = joint_name
        self.goal_constraint = goal_constraint

        if not self.get_robot().is_joint_continuous(joint_name):
            raise ConstraintException(u'{} called with non continuous joint {}'.format(self.__class__.__name__,
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

        # weight = self.magic_weight_function(w.Abs(error),
        #                                     0.0, WEIGHTS[5],
        #                                     np.pi / 8, WEIGHTS[4],
        #                                     np.pi / 6, WEIGHTS[3],
        #                                     np.pi / 4, WEIGHTS[1])

        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint('',
                            lower=capped_err,
                            upper=capped_err,
                            weight=weight,
                            expression=current_joint,
                            goal_constraint=self.goal_constraint)

    def __str__(self):
        s = super(JointPositionContinuous, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionPrismatic(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, joint_name, goal, weight=WEIGHT_BELOW_CA, max_velocity=4535, max_acceleration=0.1,
                 goal_constraint=False):
        """
        This goal will move a prismatic joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, m/s, default 4535, meaning the urdf/config limits are active
        """
        # TODO add goal constraint
        super(JointPositionPrismatic, self).__init__(god_map)
        self.joint_name = joint_name
        self.goal_constraint = goal_constraint
        if not self.get_robot().is_joint_prismatic(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__class__.__name__,
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
        # weight = self.magic_weight_function(w.Abs(err),
        #                                     0.0, WEIGHTS[5],
        #                                     0.01, WEIGHTS[4],
        #                                     0.05, WEIGHTS[3],
        #                                     0.06, WEIGHTS[1])
        # capped_err = self.limit_acceleration(current_joint, err, max_acceleration, max_velocity)
        capped_err = self.limit_velocity(err, max_velocity)

        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint('',
                            lower=capped_err,
                            upper=capped_err,
                            weight=weight,
                            expression=current_joint,
                            goal_constraint=self.goal_constraint)

    def __str__(self):
        s = super(JointPositionPrismatic, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionRevolute(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, joint_name, goal, weight=WEIGHT_BELOW_CA, max_velocity=3451, max_acceleration=1,
                 goal_constraint=True):
        """
        This goal will move a revolute joint to the goal position
        :param joint_name: str
        :param goal: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        super(JointPositionRevolute, self).__init__(god_map)
        self.joint_name = joint_name
        self.goal_constraint = goal_constraint
        if not self.get_robot().is_joint_revolute(joint_name):
            raise ConstraintException(u'{} called with non revolute joint {}'.format(self.__class__.__name__,
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
        # capped_err = self.limit_acceleration(current_joint, err, max_acceleration, max_velocity)
        capped_err = self.limit_velocity(err, max_velocity)

        # weight = self.magic_weight_function(w.Abs(err),
        #                                     0.0, WEIGHTS[5],
        #                                     np.pi / 8, WEIGHTS[4],
        #                                     np.pi / 6, WEIGHTS[3],
        #                                     np.pi / 4, WEIGHTS[1])
        # weight = WEIGHTS[5]
        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint('',
                            lower=capped_err,
                            upper=capped_err,
                            weight=weight,
                            expression=current_joint,
                            goal_constraint=self.goal_constraint)

    def __str__(self):
        s = super(JointPositionRevolute, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class Shaking(Constraint):
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'
    frequency = u'frequency'

    def __init__(self, god_map, joint_name, frequency, weight=WEIGHT_ABOVE_CA, max_velocity=3451,
                 max_acceleration=1, goal_constraint=True):
        """
        This constraint will shake a joint with the given frequency
        :param joint_name: str
        :param frequency: float
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_velocity: float, rad/s, default 3451, meaning the urdf/config limits are active
        """
        super(Shaking, self).__init__(god_map)
        self.joint_name = joint_name
        self.goal_constraint = goal_constraint

        params = {self.frequency: frequency,
                  self.weight: weight,
                  self.max_velocity: max_velocity,
                  self.max_acceleration: max_acceleration}
        self.save_params_on_god_map(params)

    def make_constraints(self):
        """
        example:
        name=u'Shaking'
        parameter_value_pair='{
            "joint_name": "torso_lift_joint", #required
            "frequency"=5, #required
            "weight": 1, #optional
            "max_velocity": 1 #optional -- rad/s or m/s depending on joint; can not go higher than urdf limit
        }'
        :return:
        """
        current_joint = self.get_input_joint_position(self.joint_name)
        frequency = self.get_input_float(self.frequency)
        weight = self.get_input_float(self.weight)
        max_velocity = w.Min(self.get_input_float(self.max_velocity),
                             self.get_robot().get_joint_velocity_limit_expr(self.joint_name))
        time = self.get_god_map().to_symbol(identifier.time)
        time_in_secs = self.get_input_sampling_period() * time

        fun_params = frequency * 2.0 * w.pi * time_in_secs
        err = w.cos(fun_params)
        capped_err = self.limit_velocity(err, max_velocity)
        weight = self.normalize_weight(max_velocity, weight)
        self.add_constraint(u'',
                            lower=capped_err,
                            upper=capped_err,
                            weight=weight,
                            expression=current_joint,
                            goal_constraint=self.goal_constraint)

    def __str__(self):
        s = super(Shaking, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class AvoidJointLimitsRevolute(Constraint):
    goal = u'goal'
    weight_id = u'weight'
    max_velocity = u'max_velocity'
    percentage = u'percentage'

    def __init__(self, god_map, joint_name, weight=0.1, max_linear_velocity=1e9, percentage=5):
        """
        This goal will push revolute joints away from their position limits
        :param joint_name: str
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_linear_velocity: float, default 1e9, meaning the urdf/config limit will kick in
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        super(AvoidJointLimitsRevolute, self).__init__(god_map)
        self.joint_name = joint_name
        if not self.get_robot().is_joint_revolute(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__class__.__name__,
                                                                                      joint_name))

        params = {self.weight_id: weight,
                  self.max_velocity: max_linear_velocity,
                  self.percentage: percentage}
        self.save_params_on_god_map(params)

    def make_constraints(self):
        weight = self.get_input_float(self.weight_id)
        joint_symbol = self.get_input_joint_position(self.joint_name)
        percentage = self.get_input_float(self.percentage) / 100.
        lower_limit, upper_limit = self.get_robot().get_joint_limits(self.joint_name)
        max_velocity = w.Min(self.get_input_float(self.max_velocity),
                             self.get_robot().get_joint_velocity_limit_expr(self.joint_name))

        joint_range = upper_limit - lower_limit
        center = (upper_limit + lower_limit) / 2.

        current_joint = self.get_input_joint_position(self.joint_name)
        max_error = joint_range / 2. * percentage

        upper_goal = center + joint_range / 2. * (1 - percentage)
        lower_goal = center - joint_range / 2. * (1 - percentage)

        upper_err = upper_goal - current_joint
        lower_err = lower_goal - current_joint

        upper_err_capped = self.limit_velocity(upper_err, max_velocity)
        lower_err_capped = self.limit_velocity(lower_err, max_velocity)

        error = w.Max(w.Abs(w.Min(upper_err, 0)), w.Abs(w.Max(lower_err, 0)))
        weight = weight * (error / max_error)

        weight_normalized = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'',
                            lower=lower_err_capped,
                            upper=upper_err_capped,
                            weight=weight_normalized,
                            expression=joint_symbol,
                            goal_constraint=False)

    def __str__(self):
        s = super(AvoidJointLimitsRevolute, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class AvoidJointLimitsPrismatic(Constraint):
    goal = u'goal'
    weight_id = u'weight'
    max_velocity = u'max_velocity'
    percentage = u'percentage'

    def __init__(self, god_map, joint_name, weight=0.1, max_angular_velocity=1e9, percentage=5):
        """
        This goal will push prismatic joints away from their position limits
        :param joint_name: str
        :param weight: float, default WEIGHT_BELOW_CA
        :param max_angular_velocity: float, default 1e9, meaning the urdf/config limit will kick in
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        """
        super(AvoidJointLimitsPrismatic, self).__init__(god_map)
        self.joint_name = joint_name
        if not self.get_robot().is_joint_prismatic(joint_name):
            raise ConstraintException(u'{} called with non prismatic joint {}'.format(self.__class__.__name__,
                                                                                      joint_name))

        params = {self.weight_id: weight,
                  self.max_velocity: max_angular_velocity,
                  self.percentage: percentage, }
        self.save_params_on_god_map(params)

    def make_constraints(self):
        weight = self.get_input_float(self.weight_id)
        joint_symbol = self.get_input_joint_position(self.joint_name)
        percentage = self.get_input_float(self.percentage) / 100.
        lower_limit, upper_limit = self.get_robot().get_joint_limits(self.joint_name)
        max_velocity = w.Min(self.get_input_float(self.max_velocity),
                             self.get_robot().get_joint_velocity_limit_expr(self.joint_name))

        joint_range = upper_limit - lower_limit
        center = (upper_limit + lower_limit) / 2.

        current_joint = self.get_input_joint_position(self.joint_name)
        max_error = joint_range / 2. * percentage

        upper_goal = center + joint_range / 2. * (1 - percentage)
        lower_goal = center - joint_range / 2. * (1 - percentage)

        upper_err = upper_goal - current_joint
        lower_err = lower_goal - current_joint

        upper_err_capped = self.limit_velocity(upper_err, max_velocity)
        lower_err_capped = self.limit_velocity(lower_err, max_velocity)

        error = w.Max(w.Abs(w.Min(upper_err, 0)), w.Abs(w.Max(lower_err, 0)))
        weight = weight * (error / max_error)

        weight_normalized = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'',
                            lower=lower_err_capped,
                            upper=upper_err_capped,
                            weight=weight_normalized,
                            expression=joint_symbol,
                            goal_constraint=False)

    def __str__(self):
        s = super(AvoidJointLimitsPrismatic, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class JointPositionList(Constraint):
    def __init__(self, god_map, goal_state, weight=None, max_velocity=None, goal_constraint=None):
        """
        This goal takes a joint state and adds the other JointPosition goals depending on their type
        :param goal_state: JointState as json
        :param weight: float, default is the default of the added joint goals
        :param max_velocity: float, default is the default of the added joint goals
        """
        super(JointPositionList, self).__init__(god_map)
        self.constraints = []
        for i, joint_name in enumerate(goal_state[u'name']):
            if not self.get_robot().has_joint(joint_name):
                raise KeyError(u'unknown joint "{}"'.format(joint_name))
            goal_position = goal_state[u'position'][i]
            params = {u'joint_name': joint_name,
                      u'goal': goal_position}
            if weight is not None:
                params[u'weight'] = weight
            if max_velocity is not None:
                params[u'max_velocity'] = max_velocity
            if goal_constraint is not None:
                params[u'goal_constraint'] = goal_constraint
            if self.get_robot().is_joint_continuous(joint_name):
                self.constraints.append(JointPositionContinuous(god_map, **params))
            elif self.get_robot().is_joint_revolute(joint_name):
                self.constraints.append(JointPositionRevolute(god_map, **params))
            elif self.get_robot().is_joint_prismatic(joint_name):
                self.constraints.append(JointPositionPrismatic(god_map, **params))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())


class AvoidJointLimits(Constraint):
    def __init__(self, god_map, percentage=15, weight=WEIGHT_BELOW_CA):
        """
        This goal will push joints away from their position limits
        :param percentage: float, default 15, if limits are 0-100, the constraint will push into the 15-85 range
        :param weight: float, default WEIGHT_BELOW_CA
        """
        super(AvoidJointLimits, self).__init__(god_map)
        self.constraints = []
        for joint_name in self.get_robot().controlled_joints:
            if self.get_robot().is_joint_revolute(joint_name):
                self.constraints.append(AvoidJointLimitsRevolute(god_map,
                                                                 joint_name=joint_name,
                                                                 percentage=percentage,
                                                                 weight=weight))
            elif self.get_robot().is_joint_prismatic(joint_name):
                self.constraints.append(AvoidJointLimitsPrismatic(god_map,
                                                                  joint_name=joint_name,
                                                                  percentage=percentage,
                                                                  weight=weight))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())


class BasicCartesianConstraint(Constraint):
    goal = u'goal'
    weight = u'weight'
    max_velocity = u'max_velocity'
    max_acceleration = u'max_acceleration'

    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.1, max_acceleration=0.1,
                 weight=WEIGHT_ABOVE_CA,
                 goal_constraint=False):
        """
        dont use me
        """
        super(BasicCartesianConstraint, self).__init__(god_map)
        self.root = root_link
        self.tip = tip_link

        self.goal_constraint = goal_constraint
        goal = self.parse_and_transform_PoseStamped(goal, self.root)

        params = {self.goal: goal,
                  self.max_acceleration: max_acceleration,
                  self.max_velocity: max_velocity,
                  self.weight: weight}
        self.save_params_on_god_map(params)

    def get_goal_pose(self):
        return self.get_input_PoseStamped(self.goal)

    def __str__(self):
        s = super(BasicCartesianConstraint, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class CartesianPosition(BasicCartesianConstraint):

    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.1, max_acceleration=0.1,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to achieve a goal position for tip link
        :param root_link: str, root link of kinematic chain
        :param tip_link: str, tip link of kinematic chain
        :param goal: PoseStamped as json
        :param max_velocity: float, m/s, default 0.1
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPosition, self).__init__(god_map, root_link, tip_link, goal, max_velocity, max_acceleration,
                                                weight, goal_constraint)

    def make_constraints(self):
        r_P_g = w.position_of(self.get_goal_pose())
        max_velocity = self.get_input_float(self.max_velocity)
        max_acceleration = self.get_input_float(self.max_acceleration)
        weight = self.get_input_float(self.weight)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root, self.tip,
                                               self.goal_constraint, weight)


class CartesianPositionStraight(BasicCartesianConstraint):
    start = u'start'

    def __init__(self, god_map, root_link, tip_link, goal,
                 max_velocity=0.1,
                 max_acceleration=0.1,
                 weight=WEIGHT_ABOVE_CA,
                 goal_constraint=True):
        super(CartesianPositionStraight, self).__init__(god_map,
                                                        root_link,
                                                        tip_link,
                                                        goal,
                                                        max_velocity,
                                                        max_acceleration,
                                                        weight,
                                                        goal_constraint)

        start = tf.lookup_pose(self.root, self.tip)

        params = {self.start: start}
        self.save_params_on_god_map(params)

    def get_tip_pose(self):
        return self.get_input_PoseStamped(self.tip)

    def make_constraints(self):
        """
        example:
        name='CartesianPositionStraight'
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
        root_P_goal = w.position_of(self.get_goal_pose())
        root_P_tip = w.position_of(self.get_fk(self.root, self.tip))
        root_V_start = w.position_of(self.get_input_PoseStamped(self.start))
        max_velocity = self.get_input_float(self.max_velocity)
        max_acceleration = self.get_input_float(self.max_acceleration)
        weight = self.get_input_float(self.weight)

        # Constraint to go to goal pos
        self.add_minimize_position_constraints(root_P_goal,
                                               max_velocity,
                                               max_acceleration,
                                               self.root,
                                               self.tip,
                                               self.goal_constraint,
                                               weight,
                                               prefix=u'goal')

        # self.add_debug_vector(u'start_point', root_P_goal)
        dist, nearest = w.distance_point_to_line_segment(root_P_tip,
                                                         root_V_start,
                                                         root_P_goal)
        # Constraint to stick to the line
        self.add_minimize_position_constraints(r_P_g=nearest,
                                               max_velocity=max_velocity,
                                               max_acceleration=max_acceleration,
                                               root=self.root,
                                               tip=self.tip,
                                               goal_constraint=self.goal_constraint,
                                               prefix=u'line',
                                               weight=WEIGHT_ABOVE_CA)


class CartesianVelocityLimit(Constraint):
    goal = u'goal'
    weight_id = u'weight'
    max_linear_velocity_id = u'max_linear_velocity'
    max_angular_velocity_id = u'max_angular_velocity'
    percentage = u'percentage'

    def __init__(self, god_map, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, hard=True):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param weight: float, default WEIGHT_ABOVE_CA
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param hard: bool, default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        """
        super(CartesianVelocityLimit, self).__init__(god_map)
        self.root_link = root_link
        self.tip_link = tip_link
        self.hard = hard

        params = {self.weight_id: weight,
                  self.max_linear_velocity_id: max_linear_velocity,
                  self.max_angular_velocity_id: max_angular_velocity
                  }
        self.save_params_on_god_map(params)

    def make_constraints(self):
        weight = self.get_input_float(self.weight_id)
        max_linear_velocity = self.get_input_float(self.max_linear_velocity_id)
        max_angular_velocity = self.get_input_float(self.max_angular_velocity_id)
        sample_period = self.get_input_sampling_period()

        root_T_tip = self.get_fk(self.root_link, self.tip_link)
        tip_evaluated_T_root = self.get_fk_evaluated(self.tip_link, self.root_link)
        root_P_tip = w.position_of(root_T_tip)

        linear_weight = self.normalize_weight(max_linear_velocity, weight)

        if self.hard:
            slack_limit = 0
        else:
            slack_limit = 1e9

        self.add_constraint(u'/linear/x',
                            lower=-max_linear_velocity * sample_period,
                            upper=max_linear_velocity * sample_period,
                            weight=linear_weight,
                            expression=root_P_tip[0],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit)
        self.add_constraint(u'/linear/y',
                            lower=-max_linear_velocity * sample_period,
                            upper=max_linear_velocity * sample_period,
                            weight=linear_weight,
                            expression=root_P_tip[1],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )
        self.add_constraint(u'/linear/z',
                            lower=-max_linear_velocity * sample_period,
                            upper=max_linear_velocity * sample_period,
                            weight=linear_weight,
                            expression=root_P_tip[2],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

        root_R_tip = w.rotation_of(root_T_tip)
        tip_evaluated_R_root = w.rotation_of(tip_evaluated_T_root)

        hack = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

        axis, angle = w.axis_angle_from_matrix(w.dot(w.dot(tip_evaluated_R_root, hack), root_R_tip))
        angular_weight = self.normalize_weight(max_angular_velocity, weight)

        axis_angle = axis * angle

        self.add_constraint(u'/angular/x',
                            lower=-max_angular_velocity * sample_period,
                            upper=max_angular_velocity * sample_period,
                            weight=angular_weight,
                            expression=axis_angle[0],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

        self.add_constraint(u'/angular/y',
                            lower=-max_angular_velocity * sample_period,
                            upper=max_angular_velocity * sample_period,
                            weight=angular_weight,
                            expression=axis_angle[1],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

        self.add_constraint(u'/angular/z',
                            lower=-max_angular_velocity * sample_period,
                            upper=max_angular_velocity * sample_period,
                            weight=angular_weight,
                            expression=axis_angle[2],
                            goal_constraint=False,
                            lower_slack_limit=-slack_limit,
                            upper_slack_limit=slack_limit
                            )

    def __str__(self):
        s = super(CartesianVelocityLimit, self).__str__()
        return u'{}/{}/{}'.format(s, self.root_link, self.tip_link)


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
#         trans_scale = w.Min(trans_error, max_velocity * t)
#         trans_control = w.save_division(trans_error_vector, trans_error) * trans_scale
#
#         self.add_constraint(u'x', lower=trans_control[0],
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
#         trans_scale = w.Min(trans_error, max_velocity * t)
#         trans_control = w.save_division(trans_error_vector, trans_error) * trans_scale
#
#         self.add_constraint(u'y', lower=trans_control[1],
#                             upper=trans_control[1],
#                             weight=weight,
#                             expression=current_position[1])


class CartesianOrientation(BasicCartesianConstraint):
    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.5, max_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=True):
        """
        Don't use me
        """
        super(CartesianOrientation, self).__init__(god_map=god_map,
                                                   root_link=root_link,
                                                   tip_link=tip_link,
                                                   goal=goal,
                                                   max_velocity=max_velocity,
                                                   max_acceleration=max_acceleration,
                                                   weight=weight,
                                                   goal_constraint=goal_constraint)

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
        weight = self.get_input_float(self.weight)

        current_rotation = w.rotation_of(self.get_fk(self.root, self.tip))
        current_evaluated_rotation = w.rotation_of(self.get_fk_evaluated(self.root, self.tip))

        hack = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)

        axis, current_angle = w.axis_angle_from_matrix(
            w.dot(w.dot(current_evaluated_rotation.T, hack), current_rotation))
        c_aa = (axis * current_angle)

        axis, angle = w.axis_angle_from_matrix(w.dot(current_rotation.T, goal_rotation))

        capped_angle = self.limit_acceleration(current_angle,
                                               angle,
                                               max_acceleration,
                                               max_velocity)

        r_rot_control = axis * capped_angle

        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'/0', lower=r_rot_control[0],
                            upper=r_rot_control[0],
                            weight=weight,
                            expression=c_aa[0],
                            goal_constraint=self.goal_constraint)
        self.add_constraint(u'/1', lower=r_rot_control[1],
                            upper=r_rot_control[1],
                            weight=weight,
                            expression=c_aa[1],
                            goal_constraint=self.goal_constraint)
        self.add_constraint(u'/2', lower=r_rot_control[2],
                            upper=r_rot_control[2],
                            weight=weight,
                            expression=c_aa[2],
                            goal_constraint=self.goal_constraint)


class CartesianOrientationSlerp(BasicCartesianConstraint):
    def __init__(self, god_map, root_link, tip_link, goal, max_velocity=0.5, max_accleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will the kinematic chain from root_link to tip_link to achieve a rotation goal for the tip link
        :param root_link: str, root link of the kinematic chain
        :param tip_link: str, tip link of the kinematic chain
        :param goal: PoseStamped as json
        :param max_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianOrientationSlerp, self).__init__(god_map=god_map,
                                                        root_link=root_link,
                                                        tip_link=tip_link,
                                                        goal=goal,
                                                        max_velocity=max_velocity,
                                                        max_acceleration=max_accleration,
                                                        weight=weight,
                                                        goal_constraint=goal_constraint)

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

        self.add_minimize_rotation_constraints(r_R_g, self.root, self.tip, max_velocity, weight, self.goal_constraint)


class CartesianPose(Constraint):
    def __init__(self, god_map, root_link, tip_link, goal, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, weight=.0, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPose, self).__init__(god_map)
        self.constraints = []
        self.constraints.append(CartesianPosition(god_map=god_map,
                                                  root_link=root_link,
                                                  tip_link=tip_link,
                                                  goal=goal,
                                                  max_velocity=max_linear_velocity,
                                                  weight=weight,
                                                  goal_constraint=goal_constraint))
        self.constraints.append(CartesianOrientationSlerp(god_map=god_map,
                                                          root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=max_angular_velocity,
                                                          weight=weight,
                                                          goal_constraint=goal_constraint))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())


class CartesianPoseStraight(Constraint):
    def __init__(self, god_map, root_link, tip_link, goal, translation_max_velocity=0.1,
                 translation_max_acceleration=0.1, rotation_max_velocity=0.5, rotation_max_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=True):
        super(CartesianPoseStraight, self).__init__(god_map)
        self.constraints = []
        self.constraints.append(CartesianPositionStraight(god_map=god_map,
                                                          root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=translation_max_velocity,
                                                          max_acceleration=translation_max_acceleration,
                                                          weight=weight,
                                                          goal_constraint=goal_constraint))
        self.constraints.append(CartesianOrientationSlerp(god_map=god_map,
                                                          root_link=root_link,
                                                          tip_link=tip_link,
                                                          goal=goal,
                                                          max_velocity=rotation_max_velocity,
                                                          max_accleration=rotation_max_acceleration,
                                                          weight=weight,
                                                          goal_constraint=goal_constraint))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())

#
#
#
#
#

class CartesianPathCarrot(Constraint):
    # TODO: Include time
    weight = u'weight'
    terminal_goal = u'terminal_goal'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathCarrot, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)

        params = {
            self.terminal_goal: self.parse_and_transform_PoseStamped(goals[-1], self.root_link),
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        params = self.setup_goal_params(params, goals)
        self.save_params_on_god_map(params)

    def setup_goal_params(self, params, goals):
        goal_str = []
        next_goal_strings = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], self.root_link)
            goal_str.append('goal_' + str(i))#w.ca.SX.sym('goal_' + str(i)))
            if i != 0:
                next_goal_strings.append('goal_' + str(i))#w.ca.SX.sym('goal_' + str(i)))

        next_goal_strings.append('goal_' + str(len(goals) - 1))#w.ca.SX.sym('goal_' + str(len(goals) - 1)))
        self.next_goal_strings = next_goal_strings # w.Matrix(next_goal_strings)
        self.goal_strings = goal_str #w.Matrix(goal_str)
        return params

    def get_goal_expr(self):
        """Will return the next normal with a higher normal time as the normal time calculated from the current position."""
        # todo: clean the mess below
        curr_normals, curr_normal_times = self.get_normals(self.goal_strings, self.next_goal_strings,
                                                           w.position_of(self.get_fk(self.root_link, self.tip_link)))
        curr_normal_dists = self.get_normal_dists(curr_normals)
        zero_one_mapping = self.zero_one_mapping_if_equal(curr_normal_dists, self.trajectory_length, w.ca.mmin(curr_normal_dists))
        curr_normal_time = self.select(curr_normal_times, zero_one_mapping)
        next_normals, next_normal_times = self.get_normals(self.goal_strings, self.next_goal_strings, self.predict())
        zero_one_mapping_n = self.zero_one_mapping_if_greater(next_normal_times, self.trajectory_length, curr_normal_time)
        next_normals_closer_to_goal = next_normals * zero_one_mapping_n
        next_normal_dists = self.get_normal_dists(next_normals_closer_to_goal)
        zero_one_mapping_one = self.zero_one_mapping_if_equal(next_normal_dists, self.trajectory_length,
                                                              w.ca.mmin(next_normal_dists))
        return self.select(next_normals_closer_to_goal, zero_one_mapping_one)

    def predict(self):
        v = self.get_fk_velocity(self.root_link, self.tip_link)[0:3]
        v_p = w.save_division(v, w.norm(v), 0) * self.get_input_float(self.max_linear_velocity) * 10 #todo: make parameter for num
        p = w.position_of(self.get_fk(self.root_link, self.tip_link))#self.get_robot_rTt())
        s = self.get_input_sampling_period()
        n_p = p[0:3] + v_p * s
        return w.point3(n_p[0], n_p[1], n_p[2])

    def get_normal_time(self, n, a, b):
        """
        Will return the normal time for a given normal point n between the start point a and b.
        First the normal time depends on the place in the trajectory. If a (the start point of the given
        trajectory part) is at the end of the trajectory the time is higher. The normal time can therefore
        be in [0, len(self.trajectory_length)]. If n is not a or b, the normalized distance from a to n will
        be added on the normal time. This results in the following formulation:
        normal_time = trajectory[a].index() + norm(n-a)/norm(b-a)
        """
        ps = []
        for i in range(0, self.trajectory_length):
            ps.append(w.position_of(self.get_input_PoseStamped(self.goal_strings[i])))
        m = self.zero_one_mapping_if_equal(w.Matrix(ps), self.trajectory_length, w.ca.transpose(a))
        g_i = self.select(w.Matrix([i for i in range(0, self.trajectory_length)]), m)
        n_t = w.save_division(w.norm(n - a), w.norm(b - a))
        return g_i + n_t

    def get_normal(self, p, a, b):
        """:rtype: w.point3"""
        ap = p - a
        ab = b - a
        ab = w.save_division(ab, w.norm(ab), 0)
        ab = ab * (w.dot(casadi.transpose(ap), ab))
        normal = a + ab
        normal = w.if_less(normal[0], w.Min(a[0], b[0]), b, normal)
        normal = w.if_greater(normal[0], w.Max(a[0], b[0]), b, normal)
        return normal

    def get_normals(self, goal_strings, next_goal_strings, pos):

        trajectory_len = self.trajectory_length
        normals = []
        normal_times = []

        for i in range(0, trajectory_len):
            a = w.position_of(self.get_input_PoseStamped(goal_strings[i]))
            b = w.position_of(self.get_input_PoseStamped(next_goal_strings[i]))
            n = self.get_normal(pos, a, b)
            n_t = self.get_normal_time(n, a, b)
            normals.append(n)
            normal_times.append(n_t)

        return w.Matrix(normals), w.Matrix(normal_times)

    def get_normal_dists(self, normals):

        trajectory_len = self.trajectory_length
        next_pos = self.predict()
        normal_dist_funs = []

        for i in range(0, trajectory_len):
            normal_dist_funs.append(w.norm(w.ca.transpose(normals[i,:]) - next_pos))

        return w.Matrix(normal_dist_funs)

    def zero_one_mapping_if_equal(self, l, l_len, elem):
        """
        Will take 2d array l and will compare elementwise each 1d entry of l with the given array elem.
        The compare function is here equal(). Because of the elementwise comparison, each positive
        comparison will be summed with 1/l.shape[1]. If the sum is 1 all elements were equal.
        """

        zeros_and_ones = []

        for i in range(0, l_len):
            zeros_and_ones.append(w.if_eq(1, w.Sum(w.if_eq(l[i,:], elem, 1/l.shape[1], 0)), 1, 0))

        return w.Matrix(zeros_and_ones)

    def zero_one_mapping_if_greater(self, l, l_len, elem):
        """
        Will take 2d array l and will compare elementwise each 1d entry of l with the given array elem.
        The compare function is here greater(). Because of the elementwise comparison, each positive
        comparison will be summed with the 1/l.shape[1]. If the sum is 1 all entries in l were greater.
        """

        zeros_and_ones = []

        for i in range(0, l_len):
            zeros_and_ones.append(w.if_eq(1, w.Sum(w.if_greater(l[i,:], elem, 1/l.shape[1], 0)), 1, 0))

        return w.Matrix(zeros_and_ones)

    def select(self, arr, zero_one_mapping):

        arr_and_zeros = arr * zero_one_mapping
        selected = w.sum_row(arr_and_zeros)
        return w.ca.transpose(selected)

    def make_constraints(self):

        goal_translation = self.get_goal_expr()
        self.add_debug_constraint("debugGoalX", goal_translation[0])
        self.add_debug_constraint("debugGoalY", goal_translation[1])
        self.add_debug_constraint("debugGoalZ", goal_translation[2])
        self.add_debug_constraint("debugGoalW", goal_translation[3])
        self.add_debug_constraint("debugCurrentX", w.position_of(self.get_fk(self.root_link, self.tip_link))[0])
        self.add_debug_constraint("debugCurrentY", w.position_of(self.get_fk(self.root_link, self.tip_link))[1])
        self.add_debug_constraint("debugCurrentZ", w.position_of(self.get_fk(self.root_link, self.tip_link))[2])
        self.add_debug_constraint("debugNextX", self.predict()[0])
        self.add_debug_constraint("debugNextY", self.predict()[1])
        self.add_debug_constraint("debugNextZ", self.predict()[2])

        self.minimize_position(goal_translation, self.get_weight())

    def get_closest_traj_point(self):

        current_pose = self.get_fk(self.root_link, self.tip_link)
        trajectory_len = self.trajectory_length
        dists_l = []
        inds_l = []

        for i in range(0, trajectory_len):
            n = w.norm(w.position_of(current_pose - self.get_input_PoseStamped(self.goal_strings[i])))
            dists_l.append(n)
            inds_l.append(i)

        dists = w.Matrix(dists_l)
        inds = w.Matrix(inds_l)
        dist_min = w.ca.mmin(dists)
        mapping = self.zero_one_mapping_if_equal(dists, trajectory_len, dist_min)
        return w.sum_row(inds * mapping)

    def get_weight(self):
        weight = self.get_input_float(self.weight)
        dyn_weight = weight - weight * self.get_traversed_trajectory_mult()
        return dyn_weight

    def get_traversed_trajectory_mult(self):
        """
        0 means the traversed trajectory is empty
        1 means we traversed the whole trajectory
        """
        traj_point = self.get_closest_traj_point() + 1
        return traj_point/self.trajectory_length

    def get_terminal_goal_weight_mult(self):
        """
        behaves pretty shitty

        0 means the goal is far away, 1 means the goal is close
        :rtype: float
        :returns: float in range(0,1)
        """
        terminal_goal = self.get_input_PoseStamped(self.terminal_goal)
        dis_to_goal = w.norm(w.position_of(self.get_fk(self.root_link, self.tip_link) - terminal_goal))
        distance_thresh = 1.0
        return w.if_less(dis_to_goal, distance_thresh,
                         0.5 * (distance_thresh - dis_to_goal) + 0.5, # add 0.5 as starting point from below:
                         distance_thresh / (2 * dis_to_goal)) # if dis_to_goal == distance_thresh, then
                                                              # distance_thresh / 2 * dis_to_goal == 0.5.

    def minimize_pose(self, goal, weight):
        self.minimize_position(w.position_of(goal), weight)
        self.minimize_rotation(w.rotation_of(goal), weight)

    def minimize_position(self, goal, weight):
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(goal, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(goal, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')


class CartesianPathSplineCarrot(Constraint):
    LUT_x_sym = u'LUT_x'
    LUT_y_sym = u'LUT_y'
    terminal_goal=u'terminal_goal'
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathSplineCarrot, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)
        self.calc_splines(goals)

        params = {
            self.LUT_x_sym: self.LUT_x_f,
            self.LUT_y_sym: self.LUT_y_f,
            self.terminal_goal: self.parse_and_transform_PoseStamped(goals[-1], root_link),
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        goal_str = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)
            goal_str.append('goal_' + str(i))#w.ca.SX.sym('goal_' + str(i)))

        self.goal_strings = goal_str#w.Matrix(goal_str)
        self.save_params_on_god_map(params)

    def LUT_x_f(self, ts):
        t = self.get_god_map().unsafe_get_data(identifier.time)#fixme:evtl berechnen?
        return self.LUT_x(t*ts)

    def LUT_y_f(self, ts):
        t = self.get_god_map().unsafe_get_data(identifier.time)
        return self.LUT_y(t*ts)

    def calc_splines(self, goals):
        goals_rs = [self.parse_and_transform_PoseStamped(goal, self.root_link) for goal in goals]
        xs = [0 for i in range(0, len(goals_rs))]
        ys = [0 for i in range(0, len(goals_rs))]
        for i, goal in enumerate(goals_rs):
            xs[i] = goal.pose.position.x
            ys[i] = goal.pose.position.y
        t = np.linspace(0, len(goals_rs) - 1, len(goals_rs)).astype(int).tolist()
        LUT_x = casadi.interpolant("LUT_x", "bspline", [t], xs)
        LUT_y = casadi.interpolant("LUT_y", "bspline", [t], ys)
        self.LUT_x = LUT_x
        self.LUT_y = LUT_y

    def get_closest_index(self):
        """
        :rtype: int
        """

        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        distances = []
        for i in range(0, trajectory_len):
            distances.append(self.distance_to_goal(goal_strings[i]))
        min_dis = w.ca.mmin(w.Matrix(distances))

        min_inds = []
        for i in range(0, trajectory_len):
            min_inds.append(w.if_eq(distances[i] - min_dis, 0, i, trajectory_len + 1))

        return w.ca.mmin(w.Matrix(min_inds))

    def zero_one_mapping_if_equal(self, l, l_len, elem):

        zeros_and_ones = []

        for i in range(0, l_len):
            zeros_and_ones.append(w.if_eq(l[i], elem, 1, 0))

        return w.Matrix(zeros_and_ones)

    def select(self, arr, zero_one_mapping):

        arr_and_zeros = arr * zero_one_mapping
        selected = w.sum_row(arr_and_zeros)
        return w.ca.transpose(selected)

    def get_closer_carrot_cart(self, max_dist):
        carrots_l = []
        carrots_dist_l = []
        fake_sampling_times = [0.081, 0.071, 0.061, 0.051, 0.041, 0.031, 0.021, 0.011]
        for st in fake_sampling_times:
            p = w.point3(self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_x_sym, (st,)]),
                         self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_y_sym, (st,)]),
                         0)
            p_dist = w.norm(p - w.position_of(self.get_fk(self.root_link, self.tip_link)))
            close_to_max_dist = w.if_greater_zero(p_dist - max_dist, -max_dist, p_dist - max_dist)
            carrots_l.append(p)
            carrots_dist_l.append(close_to_max_dist)
        carrots = w.Matrix(carrots_l)
        carrots_dist = w.Matrix(carrots_dist_l)
        max_close_to_max_dist = w.ca.mmax(carrots_dist)
        mapping = self.zero_one_mapping_if_equal(carrots_dist, 8, max_close_to_max_dist)
        return self.select(carrots, mapping)

    def get_carrot(self):
        closest_index = self.get_closest_index()
        closest_time = closest_index
        #goal_time = w.if_greater_eq(closest_time + 0.5, self.trajectory_length,
        #                            self.trajectory_length, closest_time + 0.5)
        x = self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_x_sym, (0.101,)])
        y = self.get_god_map().to_symbol(self.get_identifier() + [self.LUT_y_sym, (0.101,)])
        p = w.point3(x, y, 0)
        self.add_debug_constraint("dist_carrot", w.norm(p - w.position_of(self.get_fk(self.root_link, self.tip_link))))
        return p#self.get_closer_carrot_cart(0.5)

    def make_constraints(self):
        goal_translation = self.get_carrot()
        time = self.get_god_map().get_data(identifier.time)
        weight = self.get_input_float(self.weight)
        trajectory_len = self.get_input_float(self.trajectory_length)
        #dis_to_goal = self.distance_to_goal(self.terminal_goal)
        dyn_weight = weight - weight * w.if_greater(time/trajectory_len, 1, 1, time/trajectory_len)
                                        #0.3*w.if_less(dis_to_goal, 0.2, 1, w.if_less(dis_to_goal, 1, 1/30, 1/dis_to_goal)))
        self.add_debug_constraint("debugGoalX", goal_translation[0])
        self.add_debug_constraint("debugGoalY", goal_translation[1])
        self.add_debug_constraint("debugGoalZ", goal_translation[2])
        self.minimize_position(goal_translation, WEIGHT_ABOVE_CA)

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk(self.root_link, self.tip_link)))

    def minimize_pose(self, goal, weight):
        self.minimize_position(w.position_of(goal), weight)
        self.minimize_rotation(w.rotation_of(goal), weight)

    def minimize_position(self, goal, weight):
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(goal, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(goal, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')


class CartesianPathError(Constraint):
    get_next_py_f = u'get_next_py_f'
    goal = u'goal'
    goal_time = u'goal_time'
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathError, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)

        params = {
            #self.goal: self.goals_rs[0],
            #self.goal_time: 0,
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        goal_str = []
        next_goal_strings = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)
            goal_str.append(w.ca.SX.sym('goal_' + str(i)))
            if i != 0:
                next_goal_strings.append(w.ca.SX.sym('goal_' + str(i)))

        next_goal_strings.append(w.ca.SX.sym('goal_' + str(len(goals) - 1)))
        self.next_goal_strings = w.Matrix(next_goal_strings)
        self.goal_strings = w.Matrix(goal_str)
        self.save_params_on_god_map(params)

    # def make_constraints(self):
    #    self.get_god_map().to_symbol(self.get_identifier() + [self.get_next_py_f, tuple()])
    #    self.minimize_pose(self.goal, WEIGHT_COLLISION_AVOIDANCE)

    #def calc_splines(self):
    #    xs = np.zeros(len(self.goals_rs))
    #    ys = np.array(len(self.goals_rs))
    #    for i, goal in enumerate(self.goals_rs):
    #        xs[i] = goal.pose.position.x
    #        ys[i] = goal.pose.position.y
    #    self.LUT_x = casadi.interpolant("self.LUT_x", "bspline",
    #                                    np.linspace(0, len(self.goals_rs), len(self.goals_rs) - 1), xs)
    #    self.LUT_y = casadi.interpolant("self.LUT_y", "bspline",
    #                                    np.linspace(0, len(self.goals_rs), len(self.goals_rs) - 1), ys)
        # tck, u = interpolate.splprep([xs, ys], s=0.0)
        # x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)
        # self.interpolated_path = np.array([x_i, y_i])

    #def get_error(self, p, v_time):
    #    return casadi.sumsqr(w.position_of(p)[0] - self.LUT_x(v_time * time)) + \
    #           casadi.sumsqr(w.position_of(p)[1] - self.LUT_y(v_time * time))

    def get_error_to_goal(self, p):
        #next_index = w.if_greater(index + 1, self.trajectory_length, self.trajectory_length, index + 1)
        next_goal = w.position_of(self.get_input_PoseStamped(self.get_next_goal()))
        return casadi.sumsqr(w.position_of(p)[0] - next_goal[0]) + \
               casadi.sumsqr(w.position_of(p)[1] - next_goal[1])

    def get_closest_index(self):
        """
        :rtype: int
        """

        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        x = casadi.SX.sym('x', 1)
        f = casadi.Function('f', [x], [self.distance_to_goal(x)])
        f_map = f.map(trajectory_len)
        distances = f_map(goal_strings)
        min_dis = casadi.mmin(distances)

        d = casadi.SX.sym('d', 1)
        i = casadi.SX.sym('i', 1)
        g = casadi.Function('g', [d, i], [w.if_eq(d - min_dis, 0, i, trajectory_len + 1)])  # FIXME: kinda retarded
        g_map = g.map(trajectory_len)
        min_index = casadi.mmin(g_map(distances, casadi.linspace(0, trajectory_len - 1, trajectory_len)))

        return min_index

    def get_next_goal(self):
        """
        :rtype: int
        """

        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        x = casadi.SX.sym('x', 1)
        f = casadi.Function('f', [x], [self.distance_to_goal(x)])
        f_map = f.map(trajectory_len)
        distances = f_map(goal_strings)
        min_dis = casadi.mmin(distances)

        d = casadi.SX.sym('d', 1)
        i = casadi.SX.sym('i', 1)
        g = casadi.Function('g', [d, i], [w.if_eq(d - min_dis, 0, i, trajectory_len + 1)])  # FIXME: kinda retarded
        g_map = g.map(trajectory_len)
        goal_str = casadi.mmin(g_map(distances, self.next_goal_strings))

        return goal_str

    def get_error(self):
        closest_index = self.get_closest_index()
        return self.get_error_to_goal(self.get_fk(self.root_link, self.tip_link), closest_index)

    #def make_constraints(self):
    #    max_velocity = self.get_input_float(self.max_linear_velocity) #w.Min(self.get_input_float(self.max_linear_velocity),
    #                         #self.get_robot().get_joint_velocity_limit_expr(self.tip_link))
    #    weight = self.get_input_float(self.weight)
#
    #    self.add_debug_constraint("debugGoali", self.get_input_float(self.goal_time))
    #    err = self.get_error()
    #    capped_err = self.limit_velocity(err, max_velocity)
    #    weight = self.normalize_weight(max_velocity, weight)
#
    #    self.add_constraint('',
    #                        lower=capped_err,
    #                        upper=capped_err,
    #                        weight=weight,
    #                        expression=err,
    #                        goal_constraint=self.goal_constraint)

    def make_constraints(self):
        goal_str = self.get_next_goal()
        self.add_debug_constraint("debugGoali", self.get_input_float(self.goal_time))
        self.minimize_position(self.get_input_PoseStamped(goal_str), WEIGHT_COLLISION_AVOIDANCE)

    # def get_closest_index_py(self):
    #    goal_dists = list(map(lambda ps: self.distance_to_goal_py(ps), self.goals_rs))
    #    return goal_dists.index(min(goal_dists))

    #def update_goal_time(self):
    #    goal_time = self.get_input_float(self.goal_time)
    #    next_goal_time = w.if_greater(goal_time + 0.05, 1.0, 1.0, goal_time + 0.05)
    #    new_goal_time = w.if_less(self.distance_to_goal(self.get_pose(goal_time)),
    #                              self.distance_to_goal(self.get_pose(next_goal_time)),
    #                              goal_time, next_goal_time)
    #    self.get_god_map().set_data(self.get_identifier() + [self.goal_time], new_goal_time)

    #def get_next_goal(self):
    #    self.update_goal_time()
    #    goal_time = self.get_input_float(self.goal_time)
    #    next_goal_time = w.if_greater(goal_time + 0.05, 1.0, 1.0, goal_time + 0.05)
    #    return self.get_pose(next_goal_time)

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk(self.root_link, self.tip_link)))

    def minimize_pose(self, goal, weight):
        self.minimize_position(goal, weight)
        self.minimize_rotation(goal, weight)

    def minimize_position(self, goal, weight):
        r_P_g = w.position_of(goal)
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        r_R_g = w.rotation_of(goal)
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(r_R_g, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')

class CartesianPathG(Constraint):
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathG, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.trajectory_length = len(goals)

        params = {
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }

        goal_str = []
        next_goal_strings = []
        debug_goals = []
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)
            goal_str.append(w.ca.SX.sym('goal_' + str(i)))
            debug_goals.append(self.parse_and_transform_PoseStamped(goals[i], root_link))
            if i != 0:
                next_goal_strings.append(w.ca.SX.sym('goal_' + str(i)))

        next_goal_strings.append(w.ca.SX.sym('goal_' + str(len(goals) - 1)))
        self.next_goal_strings = w.Matrix(next_goal_strings)
        self.goal_strings = w.Matrix(goal_str)
        self.debug_goals = debug_goals
        self.save_params_on_god_map(params)

    def get_min_distance(self):
        trajectory_len = self.trajectory_length
        goal_strings = self.goal_strings

        x = casadi.SX.sym('x', 1)
        f = casadi.Function('f', [x], [self.distance_to_goal(x)])
        f_map = f.map(trajectory_len)
        distances = f_map(goal_strings)
        return casadi.mmin(distances), distances

    def get_next_index(self):
        """
        :rtype: int
        """

        min_dis, distances = self.get_min_distance()
        d = casadi.SX.sym('d', 1)
        i = casadi.SX.sym('i', 1)
        g = casadi.Function('g', [d, i], [w.if_eq(d - min_dis, 0, i, self.trajectory_length + 1)])  # FIXME: kinda retarded
        g_map = g.map(self.trajectory_length)
        min_index = casadi.mmin(g_map(distances, casadi.linspace(0, self.trajectory_length - 1, self.trajectory_length)))

        return w.if_greater(min_index + 1, self.trajectory_length, self.trajectory_length, min_index + 1)

    def get_next_goal(self):
        """
        :rtype: int
        """
        return self.next_goal_strings[casadi.evalf(self.get_next_index())]

    def make_constraints(self):
        goal_str = self.get_next_goal()
        d, _ = self.get_min_distance()
        p = w.position_of(self.get_fk(self.root_link, self.tip_link))
        self.add_debug_constraint("debugdist", d)
        self.add_debug_constraint("debugCurrentX", p[0])
        self.add_debug_constraint("debugCurrentY", p[1])
        self.minimize_position(self.get_input_PoseStamped(goal_str), WEIGHT_COLLISION_AVOIDANCE)

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk_evaluated(self.root_link, self.tip_link)))

    def minimize_pose(self, goal, weight):
        self.minimize_position(goal, weight)
        self.minimize_rotation(goal, weight)

    def minimize_position(self, goal, weight):
        r_P_g = w.position_of(goal)
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix='goal')

    def minimize_rotation(self, goal, weight):
        r_R_g = w.rotation_of(goal)
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(r_R_g, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix='goal')

class CartesianPathPy(Constraint):
    get_weight_py_f = u'get_weight_py_f'
    weight = u'weight'
    max_linear_velocity = u'max_linear_velocity'
    max_angular_velocity = u'max_angular_velocity'
    max_linear_acceleration = u'max_linear_acceleration'
    max_angular_acceleration = u'max_angular_acceleration'

    def __init__(self, god_map, root_link, tip_link, goals, max_linear_velocity=0.1,
                 max_angular_velocity=0.5, max_linear_acceleration=0.1, max_angular_acceleration=0.5,
                 weight=WEIGHT_ABOVE_CA, goal_constraint=False):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: str, name of the root link of the kin chain
        :param tip_link: str, name of the tip link of the kin chain
        :param goal: PoseStamped as json
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CartesianPathPy, self).__init__(god_map)
        self.constraints = []
        self.root_link = root_link
        self.tip_link = tip_link
        self.goal_constraint = goal_constraint
        self.goals_times = np.zeros(len(goals))
        self.goals_times[0] = 1.0
        self.goals_rs = [self.parse_and_transform_PoseStamped(goal, root_link) for goal in goals]
        self.robot = self.get_robot()

        params = {
            self.get_weight_py_f: self.get_weight_py,
            self.weight: weight,
            self.max_linear_velocity: max_linear_velocity,
            self.max_angular_velocity: max_angular_velocity,
            self.max_linear_acceleration: max_linear_acceleration,
            self.max_angular_acceleration: max_angular_acceleration
        }
        for i in range(0, len(goals)):
            params['goal_' + str(i)] = self.parse_and_transform_PoseStamped(goals[i], root_link)

        self.save_params_on_god_map(params)

    # def make_constraints(self):
    #    for constraint in self.constraints:
    #        self.soft_constraints.update(constraint.get_constraints())

    # @profile
    def update_goal_times(self):
        time_in_secs = time.time()

        for i, goal_time in enumerate(self.goals_times):
            if i == 0:
                continue
            if goal_time == 0:
                if self.goal_reached_py(self.goals_rs[i - 1]) == 1:
                    self.goals_times[i] = time_in_secs
                else:
                    self.goals_times[i] = 0

    def make_constraints(self):

        time = self.get_god_map().to_symbol(identifier.time)
        time_in_secs = self.get_input_sampling_period() * time
        # path = [self.goal_a, self.goal_b, self.goal_c, self.goal_d]
        #
        # # weight in [0,..., WEIGHT_BELOW_CA=1, WEIGHT_C_A=10, WEIGHT_ABOVE_CA=100]
        # # Bewegung muss Kosten verursachen. weight = 0 => ignorieren von motion controller zum ziel_xyz.

        # WILL ONLY BE EVALUATED ONCE, CUZ ITS WRITTEN IN PYTHON
        for i in range(0, len(self.goals_rs)):
            self.minimize_pose('goal_' + str(i), self.get_god_map().to_symbol(
                self.get_identifier() + [self.get_weight_py_f, tuple('goal_' + str(i))]))

        self.add_debug_constraint("debugTime", time)
        self.add_debug_constraint("debugTimeInSecs", time_in_secs)

        self.add_debug_constraint("debugweightATime", self.goals_times[0])
        self.add_debug_constraint("debugweightBTime", self.goals_times[1])

        # self.add_debug_constraint("debugweightA", self.get_weight(self.goal_a, self.goals_times[0], self.goals_times[1]))
        # self.add_debug_constraint("debugActivation_funA", self.activation_fun(self.goals_times[0]))
        self.add_debug_constraint("debugDistanceToA", self.distance_to_goal('goal_0'))

        # self.add_debug_constraint("debugweightB", self.get_weight(self.goal_b, self.goals_times[1], self.goals_times[2]))
        # self.add_debug_constraint("debugActivation_funB", self.activation_fun(self.goals_times[1]))
        self.add_debug_constraint("debugDistanceToB", self.distance_to_goal('goal_1'))

    def activation_fun(self, x):
        # time = self.get_god_map().to_symbol(identifier.time)
        # time_in_secs = self.get_input_sampling_period() * time
        ret = 2 / (1 + w.ca.exp(-x)) - 1
        return w.if_greater(ret, 0, ret, 0)

    def activation_fun_py(self, x):
        ret = 2 / (1 + np.exp(-x)) - 1
        return ret if ret > 0 else 0

    def get_weight(self, p, p_time, p_next_time):
        # p_reached = w.if_greater(p_next_time, p_time, 1, 0)
        # invalid_p_time = w.if_eq(p_time, -1, 1, 0)
        return self.activation_fun(
            p_time - p_next_time)  # w.if_greater_eq(p_reached + invalid_p_time, 1, 0, self.activation_fun(p_time))

    def get_weight_py(self, *p):
        self.update_goal_times()
        p_str = str(p).replace('\'', '').replace(',', '').replace('(', '').replace(')', '').replace(' ', '').replace(
            'u', '')
        for i in range(0, len(self.goals_rs)):
            if 'goal_' + str(i) in p_str:
                if i == len(self.goals_rs) - 1:
                    return self.activation_fun_py(self.goals_times[i])
                else:
                    return self.activation_fun_py(self.goals_times[i] - self.goals_times[i + 1])
        return 0

    def distance_to_goal(self, p):
        return w.norm(w.position_of(self.get_input_PoseStamped(p) - self.get_fk(self.root_link, self.tip_link)))

    def distance_to_goal_py(self, p):
        cur = self.robot.get_fk_pose(self.root_link, self.tip_link).pose.position
        x = p.pose.position.x - cur.x
        y = p.pose.position.y - cur.y
        z = p.pose.position.z - cur.z
        return np.linalg.norm(np.array([x, y, z]))

    def goal_reached(self, p):
        return w.if_less(self.distance_to_goal(p), 0.1, 1, 0)

    def goal_reached_py(self, p):
        return 1 if self.distance_to_goal_py(p) < 0.1 else 0

    # def activation_for_p(self, p, p_time):
    #    return w.if_eq(self.goal_reached(p), 1, 0, self.activation_fun(p_time)) #self.distance_to_last_goal(point, path) #+ self.distance_to_environment(point)

    def distance_to_last_goal(self, p, p_next):
        return WEIGHT_BELOW_CA  # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[0]), 3, #w.ca.SX.sym(path[0]) breaks stuff, since god_map thinks it has it
        # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[1]), 2,
        # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[2]), 1,
        # w.if_eq(w.ca.SX.sym(p), w.ca.SX.sym(path[3]), 0, 0))))

    def distance_to_environment(self, point):
        pass

    def minimize_pose(self, goal, weight):
        self.minimize_position(goal, weight)
        self.minimize_rotation(goal, weight)

    def minimize_position(self, goal, weight):
        r_P_g = w.position_of(self.get_input_PoseStamped(goal))
        max_velocity = self.get_input_float(self.max_linear_velocity)
        max_acceleration = self.get_input_float(self.max_linear_acceleration)

        self.add_minimize_position_constraints(r_P_g, max_velocity, max_acceleration, self.root_link, self.tip_link,
                                               self.goal_constraint, weight, prefix=goal)

    def minimize_rotation(self, goal, weight):
        r_R_g = w.rotation_of(self.get_input_PoseStamped(goal))
        max_velocity = self.get_input_float(self.max_angular_velocity)
        max_acceleration = self.get_input_float(self.max_angular_acceleration)

        self.add_minimize_rotation_constraints(r_R_g, self.root_link, self.tip_link, max_velocity, weight,
                                               self.goal_constraint, prefix=goal)

    def __str__(self):
        s = super(CartesianPathPy, self).__str__()
        return u'{}'.format(s)


class ExternalCollisionAvoidance(Constraint):
    max_velocity_id = u'max_velocity'
    hard_threshold_id = u'hard_threshold'
    soft_threshold_id = u'soft_threshold'
    max_acceleration_id = u'max_acceleration'
    num_repeller_id = u'num_repeller'

    def __init__(self, god_map, link_name, max_velocity=0.1, hard_threshold=0.0, soft_threshold=0.05, idx=0,
                 num_repeller=1, max_acceleration=0.005):
        """
        Don't use me
        """
        super(ExternalCollisionAvoidance, self).__init__(god_map)
        self.link_name = link_name
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()
        self.idx = idx

        params = {self.max_velocity_id: max_velocity,
                  self.hard_threshold_id: hard_threshold,
                  self.soft_threshold_id: soft_threshold,
                  self.max_acceleration_id: max_acceleration,
                  self.num_repeller_id: num_repeller,
                  }
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

    def get_closest_point_on_b_in_root(self):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [u'get_external_collisions',
                                                              (self.link_name,),
                                                              self.idx,
                                                              u'get_position_on_b_in_root',
                                                              tuple()]).get_expression()

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions',
                                                                  (self.link_name,),
                                                                  self.idx,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def get_number_of_external_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_number_of_external_collisions',
                                                                  (self.link_name,)])

    def make_constraints(self):
        a_P_pa = self.get_closest_point_on_a_in_a()
        # r_P_pb = self.get_closest_point_on_b_in_root()
        r_V_n = self.get_contact_normal_on_b_in_root()
        actual_distance = self.get_actual_distance()
        max_velocity = self.get_input_float(self.max_velocity_id)
        # max_acceleration = self.get_input_float(self.max_acceleration_id)
        # zero_weight_distance = self.get_input_float(self.zero_weight_distance)
        hard_threshold = self.get_input_float(self.hard_threshold_id)
        soft_threshold = self.get_input_float(self.soft_threshold_id)
        # spring_threshold = soft_threshold
        # soft_threshold = soft_threshold * 0.5
        # sample_period = self.get_input_sampling_period()
        number_of_external_collisions = self.get_number_of_external_collisions()
        num_repeller = self.get_input_float(self.num_repeller_id)

        root_T_a = self.get_fk(self.robot_root, self.link_name)

        r_P_pa = w.dot(root_T_a, a_P_pa)
        r_V_pb_pa = r_P_pa  # - r_P_pb

        dist = w.dot(r_V_n.T, r_V_pb_pa)[0]

        penetration_distance = soft_threshold - actual_distance
        # spring_penetration_distance = spring_threshold - actual_distance
        lower_limit = self.limit_velocity(penetration_distance, max_velocity)
        upper_limit = 1e9

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e9,
                                   w.Max(0, lower_limit + actual_distance - hard_threshold)
                                   )

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)

        # spring_error = spring_threshold - actual_distance
        # spring_error = w.Max(spring_error, 0)

        # spring_weight = w.if_eq(spring_threshold, soft_threshold, 0,
        #                         weight * (spring_error / (spring_threshold - soft_threshold))**2)

        # weight = w.if_less_eq(actual_distance, soft_threshold, weight,
        #                       spring_weight)

        weight = self.normalize_weight(max_velocity, weight)
        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.Min(number_of_external_collisions, num_repeller))

        # weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'/position',
                            lower=lower_limit,
                            upper=upper_limit,
                            weight=weight,
                            expression=dist,
                            lower_slack_limit=-1e9,
                            upper_slack_limit=upper_slack)

    def __str__(self):
        s = super(ExternalCollisionAvoidance, self).__str__()
        return u'{}/{}/{}'.format(s, self.link_name, self.idx)


class CollisionAvoidanceHint(Constraint):
    max_velocity_id = u'max_velocity'
    threshold_id = u'threshold'
    threshold2_id = u'threshold2'
    avoidance_hint_id = u'avoidance_hint'
    weight_id = u'weight'

    def __init__(self, god_map, tip_link, avoidance_hint, object_name, object_link_name, max_linear_velocity=0.1,
                 root_link=None,
                 max_threshold=0.05, spring_threshold=None, weight=WEIGHT_ABOVE_CA):
        """
        This goal pushes the link_name in the direction of avoidance_hint, if it is closer than spring_threshold
        to body_b/link_b.
        :param tip_link: str, name of the robot link, has to have a collision body
        :param avoidance_hint: Vector3Stamped as json, direction in which the robot link will get pushed
        :param object_name: str, name of the environment object, can be the robot, e.g. kitchen
        :param object_link_name: str, name of the link of the environment object. e.g. fridge handle
        :param max_linear_velocity: float, m/s, default 0.1
        :param root_link: str, default robot root, name of the root link for the kinematic chain
        :param max_threshold: float, default 0.05, distance at which the force has reached weight
        :param spring_threshold: float, default max_threshold, need to be >= than max_threshold weight increases from
                                        sprint_threshold to max_threshold linearly, to smooth motions
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CollisionAvoidanceHint, self).__init__(god_map)
        self.link_name = tip_link
        if root_link is None:
            self.root_link = self.get_robot().get_root()
        else:
            self.root_link = root_link
        self.robot_name = self.get_robot_unsafe().get_name()
        self.key = (tip_link, object_name, object_link_name)
        self.body_b = object_name
        self.link_b = object_link_name
        self.body_b_hash = object_name.__hash__()
        self.link_b_hash = object_link_name.__hash__()

        if spring_threshold is None:
            spring_threshold = max_threshold
        else:
            spring_threshold = max(spring_threshold, max_threshold)

        added_checks = self.get_god_map().get_data(identifier.added_collision_checks)
        if tip_link in added_checks:
            added_checks[tip_link] = max(added_checks[tip_link], spring_threshold)
        else:
            added_checks[tip_link] = spring_threshold
        self.get_god_map().set_data(identifier.added_collision_checks, added_checks)

        self.avoidance_hint = self.parse_and_transform_Vector3Stamped(avoidance_hint, self.root_link, normalized=True)

        params = {self.max_velocity_id: max_linear_velocity,
                  self.threshold_id: max_threshold,
                  self.threshold2_id: spring_threshold,
                  self.avoidance_hint_id: self.avoidance_hint,
                  self.weight_id: weight,
                  }
        self.save_params_on_god_map(params)

    def get_contact_normal_on_b_in_root(self):
        return Vector3Input(self.god_map.to_symbol,
                            prefix=identifier.closest_point + [u'get_external_collisions_long_key',
                                                               self.key,
                                                               u'get_contact_normal_in_root',
                                                               tuple()]).get_expression()

    def get_closest_point_on_a_in_a(self):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [u'get_external_collisions_long_key',
                                                              self.key,
                                                              u'get_position_on_a_in_a',
                                                              tuple()]).get_expression()

    def get_closest_point_on_b_in_root(self):
        return Point3Input(self.god_map.to_symbol,
                           prefix=identifier.closest_point + [u'get_external_collisions_long_key',
                                                              self.key,
                                                              u'get_position_on_b_in_root',
                                                              tuple()]).get_expression()

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                  self.key,
                                                                  u'get_contact_distance',
                                                                  tuple()])

    def get_body_b(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                  self.key, u'get_body_b_hash', tuple()])

    def get_link_b(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_external_collisions_long_key',
                                                                  self.key, u'get_link_b_hash', tuple()])

    def make_constraints(self):
        weight = self.get_input_float(self.weight_id)
        actual_distance = self.get_actual_distance()
        max_velocity = self.get_input_float(self.max_velocity_id)
        max_threshold = self.get_input_float(self.threshold_id)
        spring_threshold = self.get_input_float(self.threshold2_id)
        body_b_hash = self.get_body_b()
        link_b_hash = self.get_link_b()
        actual_distance_capped = w.Max(actual_distance, 0)

        root_T_a = self.get_fk(self.root_link, self.link_name)

        spring_error = spring_threshold - actual_distance_capped
        spring_error = w.Max(spring_error, 0)

        spring_weight = w.if_eq(spring_threshold, max_threshold, 0,
                                weight * (spring_error / (spring_threshold - max_threshold)) ** 2)

        weight = w.if_less_eq(actual_distance, max_threshold, weight,
                              spring_weight)
        weight = w.if_eq(body_b_hash, self.body_b_hash, weight, 0)
        weight = w.if_eq(link_b_hash, self.link_b_hash, weight, 0)
        weight = self.normalize_weight(max_velocity, weight)

        root_V_avoidance_hint = self.get_input_Vector3Stamped(self.avoidance_hint_id)

        # penetration_distance = threshold - actual_distance_capped
        error_capped = self.limit_velocity(max_velocity, max_velocity)

        root_P_a = w.position_of(root_T_a)
        expr = w.dot(root_V_avoidance_hint[:3].T, root_P_a[:3])

        self.add_constraint(u'/avoidance_hint',
                            lower=error_capped,
                            upper=error_capped,
                            weight=weight,
                            expression=expr)

    def __str__(self):
        s = super(CollisionAvoidanceHint, self).__str__()
        return u'{}/{}/{}/{}'.format(s, self.link_name, self.body_b, self.link_b)


class SelfCollisionAvoidance(Constraint):
    max_velocity_id = u'max_velocity'
    hard_threshold_id = u'hard_threshold'
    soft_threshold_id = u'soft_threshold'
    max_acceleration_id = u'max_acceleration'
    root_T_link_b = u'root_T_link_b'
    link_in_chain = u'link_in_chain'
    num_repeller_id = u'num_repeller'

    def __init__(self, god_map, link_a, link_b, max_velocity=0.1, hard_threshold=0.0, soft_threshold=0.05, idx=0,
                 num_repeller=1):
        super(SelfCollisionAvoidance, self).__init__(god_map)
        self.link_a = link_a
        self.link_b = link_b
        self.robot_root = self.get_robot().get_root()
        self.robot_name = self.get_robot_unsafe().get_name()
        self.idx = idx

        params = {self.max_velocity_id: max_velocity,
                  self.hard_threshold_id: hard_threshold,
                  self.soft_threshold_id: soft_threshold,
                  self.num_repeller_id: num_repeller}
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

    def get_number_of_self_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + [u'get_number_of_self_collisions',
                                                                  (self.link_a, self.link_b)])

    def make_constraints(self):
        repel_velocity = self.get_input_float(self.max_velocity_id)
        hard_threshold = self.get_input_float(self.hard_threshold_id)
        soft_threshold = self.get_input_float(self.soft_threshold_id)
        actual_distance = self.get_actual_distance()
        number_of_self_collisions = self.get_number_of_self_collisions()
        num_repeller = self.get_input_float(self.num_repeller_id)

        b_T_a = self.get_fk(self.link_b, self.link_a)
        pb_T_b = w.inverse_frame(self.get_b_T_pb())
        a_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        pb_P_pa = w.dot(pb_T_b, b_T_a, a_P_pa)

        dist = w.dot(pb_V_n.T, pb_P_pa)[0]

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)
        weight = self.normalize_weight(repel_velocity, weight)
        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.Min(number_of_self_collisions, num_repeller))

        penetration_distance = soft_threshold - actual_distance
        lower_limit = self.limit_velocity(penetration_distance, repel_velocity)
        upper_limit = 1e9
        # slack_limit = self.limit_velocity(actual_distance, repel_velocity)

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e9,
                                   w.Max(0, lower_limit + actual_distance - hard_threshold)
                                   )

        self.add_constraint(u'/position',
                            lower=lower_limit,
                            upper=upper_limit,
                            weight=weight,
                            expression=dist,
                            goal_constraint=False,
                            lower_slack_limit=-1e9,
                            upper_slack_limit=upper_slack)

    def __str__(self):
        s = super(SelfCollisionAvoidance, self).__str__()
        return u'{}/{}/{}/{}'.format(s, self.link_a, self.link_b, self.idx)


class AlignPlanes(Constraint):
    root_normal_id = u'root_normal'
    tip_normal_id = u'tip_normal'
    max_velocity_id = u'max_velocity'
    weight_id = u'weight'

    def __init__(self, god_map, root_link, tip_link, root_normal, tip_normal,
                 max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 goal_constraint=False):
        """
        This Goal will use the kinematic chain between tip and root normal to align both
        :param root_link: str, name of the root link for the kinematic chain
        :param tip_link: str, name of the tip link for the kinematic chain
        :param tip_normal: Vector3Stamped as json, normal at the tip of the kin chain
        :param root_normal: Vector3Stamped as json, normal at the root of the kin chain
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float, default is WEIGHT_ABOVE_CA
        :param goal_constraint: bool, default False
        """
        super(AlignPlanes, self).__init__(god_map)
        self.root = root_link
        self.tip = tip_link
        self.goal_constraint = goal_constraint

        self.tip_normal = self.parse_and_transform_Vector3Stamped(tip_normal, self.tip, normalized=True)
        self.root_normal = self.parse_and_transform_Vector3Stamped(root_normal, self.root, normalized=True)

        params = {self.root_normal_id: self.root_normal,
                  self.tip_normal_id: self.tip_normal,
                  self.max_velocity_id: max_angular_velocity,
                  self.weight_id: weight}
        self.save_params_on_god_map(params)

    def __str__(self):
        s = super(AlignPlanes, self).__str__()
        return u'{}/{}/{}_X:{}_Y:{}_Z:{}'.format(s, self.root, self.tip,
                                                 self.tip_normal.vector.x,
                                                 self.tip_normal.vector.y,
                                                 self.tip_normal.vector.z)

    def get_root_normal_vector(self):
        return self.get_input_Vector3Stamped(self.root_normal_id)

    def get_tip_normal_vector(self):
        return self.get_input_Vector3Stamped(self.tip_normal_id)

    def make_constraints(self):
        max_velocity = self.get_input_float(self.max_velocity_id)
        tip_normal__tip = self.get_tip_normal_vector()
        root_normal__root = self.get_root_normal_vector()
        weight = self.get_input_float(self.weight_id)
        self.add_minimize_vector_angle_constraints(max_velocity=max_velocity,
                                                   root=self.root,
                                                   tip=self.tip,
                                                   tip_V_tip_normal=tip_normal__tip,
                                                   root_V_goal_normal=root_normal__root,
                                                   weight=weight,
                                                   goal_constraint=self.goal_constraint)


class GraspBar(Constraint):
    bar_axis_id = u'bar_axis'
    tip_grasp_axis_id = u'tip_grasp_axis'
    bar_center_id = u'bar_center'
    bar_length_id = u'bar_length'
    translation_max_velocity_id = u'translation_max_velocity'
    rotation_max_velocity_id = u'rotation_max_velocity'
    weight_id = u'weight'

    def __init__(self, god_map, root_link, tip_link, tip_grasp_axis, bar_center, bar_axis, bar_length,
                 max_linear_velocity=0.1, max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA,
                 goal_constraint=False):
        """
        This goal can be used to grasp bars. It's like a cartesian goal with some freedom along one axis.
        :param root_link: str, root link of the kin chain
        :param tip_link: str, tip link of the kin chain
        :param tip_grasp_axis: Vector3Stamped as json, this axis of the tip will be aligned with bar_axis
        :param bar_center: PointStamped as json, center of the bar
        :param bar_axis: Vector3Stamped as json, tip_grasp_axis will be aligned with this vector
        :param bar_length: float, length of the bar
        :param max_linear_velocity: float, m/s, default 0.1
        :param max_angular_velocity: float, rad/s, default 0.5
        :param weight: float default WEIGHT_ABOVE_CA
        """
        super(GraspBar, self).__init__(god_map)
        self.root = root_link
        self.tip = tip_link
        self.goal_constraint = goal_constraint

        bar_center = self.parse_and_transform_PointStamped(bar_center, self.root)

        tip_grasp_axis = self.parse_and_transform_Vector3Stamped(tip_grasp_axis, self.tip, normalized=True)

        bar_axis = self.parse_and_transform_Vector3Stamped(bar_axis, self.root, normalized=True)

        params = {self.bar_axis_id: bar_axis,
                  self.tip_grasp_axis_id: tip_grasp_axis,
                  self.bar_center_id: bar_center,
                  self.bar_length_id: bar_length,
                  self.translation_max_velocity_id: max_linear_velocity,
                  self.rotation_max_velocity_id: max_angular_velocity,
                  self.weight_id: weight}
        self.save_params_on_god_map(params)

    def __str__(self):
        s = super(GraspBar, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)

    def get_bar_axis_vector(self):
        return self.get_input_Vector3Stamped(self.bar_axis_id)

    def get_tip_grasp_axis_vector(self):
        return self.get_input_Vector3Stamped(self.tip_grasp_axis_id)

    def get_bar_center_point(self):
        return self.get_input_PointStamped(self.bar_center_id)

    def make_constraints(self):
        translation_max_velocity = self.get_input_float(self.translation_max_velocity_id)
        rotation_max_velocity = self.get_input_float(self.rotation_max_velocity_id)
        weight = self.get_input_float(self.weight_id)

        bar_length = self.get_input_float(self.bar_length_id)
        root_V_bar_axis = self.get_bar_axis_vector()
        tip_V_tip_grasp_axis = self.get_tip_grasp_axis_vector()
        root_P_bar_center = self.get_bar_center_point()

        self.add_minimize_vector_angle_constraints(max_velocity=rotation_max_velocity,
                                                   root=self.root,
                                                   tip=self.tip,
                                                   tip_V_tip_normal=tip_V_tip_grasp_axis,
                                                   root_V_goal_normal=root_V_bar_axis,
                                                   weight=weight,
                                                   goal_constraint=self.goal_constraint)

        root_P_tip = w.position_of(self.get_fk(self.root, self.tip))

        root_P_line_start = root_P_bar_center + root_V_bar_axis * bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * bar_length / 2

        dist, nearest = w.distance_point_to_line_segment(root_P_tip, root_P_line_start, root_P_line_end)

        self.add_minimize_position_constraints(r_P_g=nearest,
                                               max_velocity=translation_max_velocity,
                                               max_acceleration=0.1,
                                               root=self.root,
                                               tip=self.tip,
                                               weight=weight,
                                               goal_constraint=self.goal_constraint)


class BasePointingForward(Constraint):
    base_forward_axis_id = u'base_forward_axis'
    max_velocity = u'max_velocity'
    range_id = u'range'
    linear_velocity_threshold_id = u'linear_velocity_threshold'
    weight_id = u'weight'

    def __init__(self, god_map, base_forward_axis=None, base_footprint=None, odom=None, velocity_tip=None,
                 range=np.pi / 8, max_velocity=np.pi / 8, linear_velocity_threshold=0.02, weight=WEIGHT_BELOW_CA):
        """
        dont use
        :param god_map: ignore
        :type base_forward_axis: Vector3Stamped as json dict
        :type base_footprint: str
        :type odom: str
        :type range: float
        :type max_velocity: float
        """
        super(BasePointingForward, self).__init__(god_map)
        if odom is not None:
            self.odom = odom
        else:
            self.odom = self.get_robot().get_root()
        if base_footprint is not None:
            self.base_footprint = base_footprint
        else:
            self.base_footprint = self.get_robot().get_non_base_movement_root()
        if velocity_tip is None:
            self.velocity_tip = self.base_footprint
        else:
            self.velocity_tip = velocity_tip
        if base_forward_axis is not None:
            self.base_forward_axis = base_forward_axis
            base_forward_axis = convert_dictionary_to_ros_message(u'geometry_msgs/Vector3Stamped', base_forward_axis)
            base_forward_axis = tf.transform_vector(self.base_footprint, base_forward_axis)
            tmp = np.array([base_forward_axis.vector.x, base_forward_axis.vector.y, base_forward_axis.vector.z])
            tmp = tmp / np.linalg.norm(tmp)
            self.base_forward_axis.vector = Vector3(*tmp)
        else:
            self.base_forward_axis = Vector3Stamped()
            self.base_forward_axis.header.frame_id = self.base_footprint
            self.base_forward_axis.vector.x = 1

        params = {self.base_forward_axis_id: self.base_forward_axis,
                  self.max_velocity: max_velocity,
                  self.range_id: range,
                  self.linear_velocity_threshold_id: linear_velocity_threshold,
                  self.weight_id: weight}
        self.save_params_on_god_map(params)

    def __str__(self):
        s = super(BasePointingForward, self).__str__()
        return u'{}/{}/{}_X:{}_Y:{}_Z:{}'.format(s, self.odom, self.base_footprint,
                                                 self.base_forward_axis.vector.x,
                                                 self.base_forward_axis.vector.y,
                                                 self.base_forward_axis.vector.z)

    def get_base_forward_axis(self):
        return self.get_input_Vector3Stamped(self.base_forward_axis_id)

    def make_constraints(self):
        range = self.get_input_float(self.range_id)
        weight = self.get_input_float(self.weight_id)
        linear_velocity_threshold = self.get_input_float(self.linear_velocity_threshold_id)
        max_velocity = self.get_input_float(self.max_velocity)

        weight = self.normalize_weight(max_velocity, weight)

        odom_T_base_footprint_dot = self.get_fk_velocity(self.odom, self.velocity_tip)
        odom_V_goal = w.vector3(odom_T_base_footprint_dot[0],
                                odom_T_base_footprint_dot[1],
                                odom_T_base_footprint_dot[2])
        odom_V_goal_length_1 = w.scale(odom_V_goal, 1)

        odom_R_base_footprint = w.rotation_of(self.get_fk(self.odom, self.base_footprint))
        base_footprint_V_current = self.get_base_forward_axis()
        odom_V_base_footprint = w.dot(odom_R_base_footprint, base_footprint_V_current)

        linear_velocity = w.norm(odom_V_goal)

        error = w.acos(w.dot(odom_V_goal_length_1.T, odom_V_base_footprint)[0])
        error_limited_lb = w.if_greater_eq(linear_velocity_threshold, linear_velocity, 0,
                                           self.limit_velocity(error + range, max_velocity))
        error_limited_ub = w.if_greater_eq(linear_velocity_threshold, linear_velocity, 0,
                                           self.limit_velocity(error - range, max_velocity))
        self.add_constraint(u'/error',
                            lower=-error_limited_lb,
                            upper=-error_limited_ub,
                            weight=weight,
                            expression=error,
                            goal_constraint=False)


class GravityJoint(Constraint):
    weight = u'weight'

    # FIXME

    def __init__(self, god_map, joint_name, object_name, goal_constraint=True):
        """
        don't use me
        """
        super(GravityJoint, self).__init__(god_map)
        self.joint_name = joint_name
        self.object_name = object_name
        self.goal_constraint = goal_constraint
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

        weight = WEIGHT_BELOW_CA
        # TODO set a reasonable velocity limit
        weight = self.normalize_weight(0.1, weight)

        self.add_constraint('',
                            lower=goal_vel,  # sw.Min(goal_vel, 0),
                            upper=goal_vel,  # sw.Max(goal_vel, 0),
                            weight=weight,
                            expression=current_joint,
                            goal_constraint=self.goal_constraint)

    def __str__(self):
        s = super(GravityJoint, self).__str__()
        return u'{}/{}'.format(s, self.joint_name)


class UpdateGodMap(Constraint):

    def __init__(self, god_map, updates):
        """
        Modifies the core data structure of giskard, only used for hacks, and you know what you are doing :)
        """
        super(UpdateGodMap, self).__init__(god_map)
        self.update_god_map([], updates)

    def update_god_map(self, identifier, updates):
        if not isinstance(updates, dict):
            raise GiskardException(u'{} used incorrectly, {} not a dict or number'.format(str(self), updates))
        for member, value in updates.items():
            next_identifier = identifier + [member]
            if isinstance(value, numbers.Number) and \
                    isinstance(self.get_god_map().get_data(next_identifier), numbers.Number):
                self.get_god_map().set_data(next_identifier, value)
            else:
                self.update_god_map(next_identifier, value)


class Pointing(Constraint):
    goal_point = u'goal_point'
    pointing_axis = u'pointing_axis'
    weight_id = u'weight'

    def __init__(self, god_map, tip_link, goal_point, root_link=None, pointing_axis=None, weight=WEIGHT_BELOW_CA,
                 goal_constraint=True):
        """
        Uses the kinematic chain from root_link to tip_link to move the pointing axis, such that it points to the goal point.
        :param tip_link: str, name of the tip of the kin chain
        :param goal_point: PointStamped as json, where the pointing_axis will point towards
        :param root_link: str, name of the root of the kin chain
        :param pointing_axis: Vector3Stamped as json, default is z axis, this axis will point towards the goal_point
        :param weight: float, default WEIGHT_BELOW_CA
        """
        # always start by calling super with god map
        super(Pointing, self).__init__(god_map)

        # use this space to process your input parameters, handle defaults etc
        if root_link is None:
            self.root = self.get_robot().get_root()
        else:
            self.root = root_link
        self.tip = tip_link
        self.goal_constraint = goal_constraint

        # you receive message in json form, use these functions to turn them into the proper types and transfrom
        # them into a goal frame
        goal_point = self.parse_and_transform_PointStamped(goal_point, self.root)

        if pointing_axis is not None:
            pointing_axis = self.parse_and_transform_Vector3Stamped(pointing_axis, self.tip, normalized=True)
        else:
            pointing_axis = Vector3Stamped()
            pointing_axis.header.frame_id = self.tip
            pointing_axis.vector.z = 1

        # save everything, that you want to reference in expressions on the god map
        params = {self.goal_point: goal_point,
                  self.pointing_axis: pointing_axis,
                  self.weight_id: weight}
        self.save_params_on_god_map(params)

    # make make some convenience functions to make your code more readable
    def get_goal_point(self):
        return self.get_input_PointStamped(self.goal_point)

    def get_pointing_axis(self):
        return self.get_input_Vector3Stamped(self.pointing_axis)

    def make_constraints(self):
        # in this function, you have to create the actual constraints
        # start by creating references to your input params in the god map
        # get_input functions generally return symbols referring to god map entries
        weight = self.get_input_float(self.weight_id)
        # TODO set a reasonable velocity limit
        weight = self.normalize_weight(0.1, weight)
        root_T_tip = self.get_fk(self.root, self.tip)
        goal_point = self.get_goal_point()
        pointing_axis = self.get_pointing_axis()

        # do some math to create your expressions and limits
        # make sure to always use function from the casadi_wrapper, here imported as "w".
        # here are some rules of thumb that often make constraints more stable:
        # 1) keep the expressions as simple as possible and move the "magic" into the lower/upper limits
        # 2) don't try to minimize the number of constraints (in this example, minimizing the angle is also possible
        #       but sometimes gets unstable)
        # 3) you can't use the normal if! use e.g. "w.if_eq"
        # 4) use self.limit_velocity on your error
        # 5) giskard will calculate the derivative of "expression". so in this example, writing -diff[0] in
        #       in expression will result in the same behavior, because goal_axis is constant.
        #       This is also the reason, why lower/upper are limits for the derivative.
        goal_axis = goal_point - w.position_of(root_T_tip)
        goal_axis /= w.norm(goal_axis)  # FIXME avoid /0
        current_axis = w.dot(root_T_tip, pointing_axis)
        diff = goal_axis - current_axis

        # add constraints to the current problem, after execution, it gets cleared automatically
        self.add_constraint(
            # name of the constraint, make use to avoid name conflicts!
            u'x',
            # lower limit for the !derivative! of the expression
            lower=diff[0],
            # upper limit for the !derivative! of the expression
            upper=diff[0],
            # tells the solver how important this constraint is, if unsure, use HIGH_WEIGHT
            weight=weight,
            # symbolic expression that describes a geometric property. make sure it as a dependency on the
            # joint state. usually achieved through "get_fk"
            expression=current_axis[0],
            # describes if this constraint must be fulfilled at the end of the trajectory
            goal_constraint=self.goal_constraint)

        self.add_constraint(u'y',
                            lower=diff[1],
                            upper=diff[1],
                            weight=weight,
                            expression=current_axis[1],
                            goal_constraint=self.goal_constraint)
        self.add_constraint(u'z',
                            lower=diff[2],
                            upper=diff[2],
                            weight=weight,
                            expression=current_axis[2],
                            goal_constraint=self.goal_constraint)

    def __str__(self):
        # helps to make sure your constraint name is unique.
        s = super(Pointing, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class OpenDoor(Constraint):
    hinge_pose_id = u'hinge_frame'
    hinge_V_hinge_axis_msg_id = u'hinge_axis'
    hinge0_T_tipGoal_id = u'hinge0_T_tipGoal'
    hinge0_T_tipStartProjected_id = u'hinge0_T_tipStartProjected'
    root_T_hinge0_id = u'root_T_hinge0'
    root_T_tipGoal_id = u'root_T_tipGoal'
    hinge0_P_tipStart_norm_id = u'hinge0_P_tipStart_norm'
    weight_id = u'weight'

    def __init__(self, god_map, tip_link, object_name, object_link_name, angle_goal, root_link=None,
                 weight=WEIGHT_ABOVE_CA):
        super(OpenDoor, self).__init__(god_map)

        if root_link is None:
            self.root = self.get_robot().get_root()
        else:
            self.root = root_link
        self.tip = tip_link

        self.angle_goal = angle_goal

        self.handle_link = object_link_name
        handle_frame_id = u'iai_kitchen/' + object_link_name

        self.object_name = object_name
        environment_object = self.get_world().get_object(object_name)
        self.hinge_joint = environment_object.get_movable_parent_joint(object_link_name)
        hinge_child = environment_object.get_child_link_of_joint(self.hinge_joint)

        hinge_frame_id = u'iai_kitchen/' + hinge_child

        hinge_V_hinge_axis = kdl.Vector(*environment_object.get_joint_axis(self.hinge_joint))
        hinge_V_hinge_axis_msg = Vector3Stamped()
        hinge_V_hinge_axis_msg.header.frame_id = hinge_frame_id
        hinge_V_hinge_axis_msg.vector.x = hinge_V_hinge_axis[0]
        hinge_V_hinge_axis_msg.vector.y = hinge_V_hinge_axis[1]
        hinge_V_hinge_axis_msg.vector.z = hinge_V_hinge_axis[2]

        hingeStart_T_tipStart = tf.msg_to_kdl(tf.lookup_pose(hinge_frame_id, self.tip))

        hinge_pose = tf.lookup_pose(self.root, hinge_frame_id)

        root_T_hingeStart = tf.msg_to_kdl(hinge_pose)
        hinge_T_handle = tf.msg_to_kdl(tf.lookup_pose(hinge_frame_id, handle_frame_id))  # constant
        hinge_joint_current = environment_object.joint_state[self.hinge_joint].position

        hingeStart_P_tipStart = hingeStart_T_tipStart.p

        projection = kdl.dot(hingeStart_P_tipStart, hinge_V_hinge_axis)
        hinge0_P_tipStartProjected = hingeStart_P_tipStart - hinge_V_hinge_axis * projection

        hinge0_T_hingeCurrent = kdl.Frame(kdl.Rotation().Rot(hinge_V_hinge_axis, hinge_joint_current))
        root_T_hinge0 = root_T_hingeStart * hinge0_T_hingeCurrent.Inverse()
        root_T_handleGoal = root_T_hinge0 * kdl.Frame(
            kdl.Rotation().Rot(hinge_V_hinge_axis, angle_goal)) * hinge_T_handle

        handleStart_T_tipStart = tf.msg_to_kdl(tf.lookup_pose(handle_frame_id, self.tip))
        root_T_tipGoal = tf.kdl_to_np(root_T_handleGoal * handleStart_T_tipStart)

        hinge0_T_tipGoal = tf.kdl_to_np(hingeStart_T_tipStart)
        hinge0_T_tipStartProjected = tf.kdl_to_np(kdl.Frame(hinge0_P_tipStartProjected))

        hinge0_P_tipStart_norm = np.linalg.norm(tf.kdl_to_np(hingeStart_P_tipStart))

        params = {
            self.hinge_pose_id: hinge_pose,
            self.hinge_V_hinge_axis_msg_id: hinge_V_hinge_axis_msg,
            self.hinge0_T_tipGoal_id: hinge0_T_tipGoal,
            self.root_T_hinge0_id: tf.kdl_to_np(root_T_hinge0),
            self.hinge0_T_tipStartProjected_id: hinge0_T_tipStartProjected,
            self.root_T_tipGoal_id: root_T_tipGoal,
            self.hinge0_P_tipStart_norm_id: hinge0_P_tipStart_norm,
            self.weight_id: weight,
        }
        self.save_params_on_god_map(params)

    def get_hinge_pose(self):
        return self.get_input_PoseStamped(self.hinge_pose_id)

    def get_hinge_axis(self):
        return self.get_input_Vector3Stamped(self.hinge_V_hinge_axis_msg_id)

    def make_constraints(self):
        base_weight = self.get_input_float(self.weight_id)
        root_T_tip = self.get_fk(self.root, self.tip)
        root_T_hinge = self.get_hinge_pose()
        hinge_V_hinge_axis = self.get_hinge_axis()[:3]
        hinge_T_root = w.inverse_frame(root_T_hinge)
        root_T_tipGoal = self.get_input_np_frame(self.root_T_tipGoal_id)
        root_T_hinge0 = self.get_input_np_frame(self.root_T_hinge0_id)
        root_T_tipCurrent = self.get_fk_evaluated(self.root, self.tip)
        hinge0_R_tipGoal = w.rotation_of(self.get_input_np_frame(self.hinge0_T_tipGoal_id))
        dist_goal = self.get_input_float(self.hinge0_P_tipStart_norm_id)
        hinge0_T_tipStartProjected = self.get_input_np_frame(self.hinge0_T_tipStartProjected_id)

        self.add_minimize_position_constraints(w.position_of(root_T_tipGoal), 0.1, 0.1, self.root, self.tip, False,
                                               weight=base_weight)

        hinge_P_tip = w.position_of(w.dot(hinge_T_root, root_T_tip))[:3]

        dist_expr = w.norm(hinge_P_tip)
        weight = self.normalize_weight(0.1, base_weight)
        self.add_constraint(u'/dist',
                            dist_goal - dist_expr,
                            dist_goal - dist_expr,
                            weight,
                            dist_expr,
                            False)

        hinge0_T_tipCurrent = w.dot(w.inverse_frame(root_T_hinge0), root_T_tipCurrent)
        hinge0_P_tipStartProjected = w.position_of(hinge0_T_tipStartProjected)
        hinge0_P_tipCurrent = w.position_of(hinge0_T_tipCurrent)[:3]

        projection = w.dot(hinge0_P_tipCurrent.T, hinge_V_hinge_axis)
        hinge0_P_tipCurrentProjected = hinge0_P_tipCurrent - hinge_V_hinge_axis * projection

        current_tip_angle_projected = w.angle_between_vector(hinge0_P_tipStartProjected, hinge0_P_tipCurrentProjected)

        hinge0_T_hingeCurrent = w.rotation_matrix_from_axis_angle(hinge_V_hinge_axis, current_tip_angle_projected)

        root_T_hingeCurrent = w.dot(root_T_hinge0, hinge0_T_hingeCurrent)

        root_R_tipGoal = w.dot(root_T_hingeCurrent, hinge0_R_tipGoal)

        self.add_minimize_rotation_constraints(root_R_tipGoal, self.root, self.tip, weight=base_weight)

    def __str__(self):
        s = super(OpenDoor, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class OpenDrawer(Constraint):
    hinge_pose_id = u'hinge_frame'  # frame of the hinge TODO: is that necessary
    hinge_V_hinge_axis_msg_id = u'hinge_axis'  # axis vector of the hinge
    root_T_tip_goal_id = u'root_T_tipGoal'  # goal of the gripper tip (where to move)

    def __init__(self, god_map, tip_link, object_name, object_link_name, distance_goal, root_link=None,
                 weight=WEIGHT_ABOVE_CA):
        """
        :type tip_link: str
        :param tip_link: tip of manipulator (gripper) which is used
        :type object_name str
        :param object_name
        :type object_link_name str
        :param object_link_name handle to grasp
        :type distance_goal float
        :param distance_goal
               relative opening distance 0 = close, 1 = fully open
        :type root_link: str
        :param root_link: default is root link of robot
        """

        super(OpenDrawer, self).__init__(god_map)

        self.constraints = []  # init empty list

        # Process input parameters
        if root_link is None:
            self.root = self.get_robot().get_root()
        else:
            self.root = root_link
        self.tip = tip_link

        self.distance_goal = distance_goal

        self.handle_link = object_link_name
        handle_frame_id = u'iai_kitchen/' + object_link_name

        self.object_name = object_name
        environment_object = self.get_world().get_object(object_name)
        # Get movable joint
        self.hinge_joint = environment_object.get_movable_parent_joint(object_link_name)
        # Child of joint
        hinge_child = environment_object.get_child_link_of_joint(self.hinge_joint)

        hinge_frame_id = u'iai_kitchen/' + hinge_child

        # Get movable axis of drawer (= prismatic joint)
        hinge_drawer_axis = kdl.Vector(
            *environment_object.get_joint_axis(self.hinge_joint))
        hinge_drawer_axis_msg = Vector3Stamped()
        hinge_drawer_axis_msg.header.frame_id = hinge_frame_id
        hinge_drawer_axis_msg.vector.x = hinge_drawer_axis[0]
        hinge_drawer_axis_msg.vector.y = hinge_drawer_axis[1]
        hinge_drawer_axis_msg.vector.z = hinge_drawer_axis[2]

        # Get joint limits TODO: check of desired goal is within limits
        min_limit, max_limit = environment_object.get_joint_limits(
            self.hinge_joint)
        current_joint_pos = environment_object.joint_state[self.hinge_joint].position

        # Avoid invalid values
        if distance_goal < min_limit:
            self.distance_goal = min_limit
        if distance_goal > max_limit:
            self.distance_goal = max_limit

        hinge_frame_id = u'iai_kitchen/' + hinge_child

        # Get frame of current tip pose
        root_T_tip_current = tf.msg_to_kdl(tf.lookup_pose(self.root, tip_link))
        hinge_drawer_axis_kdl = tf.msg_to_kdl(hinge_drawer_axis_msg)  # get axis of joint
        # Get transform from hinge to root
        root_T_hinge = tf.msg_to_kdl(tf.lookup_pose(self.root, hinge_frame_id))

        # Get translation vector from current to goal position
        tip_current_V_tip_goal = hinge_drawer_axis_kdl * (self.distance_goal - current_joint_pos)

        root_V_hinge_drawer = root_T_hinge.M * tip_current_V_tip_goal  # get vector in hinge frame
        root_T_tip_goal = deepcopy(root_T_tip_current)  # copy object to manipulate it
        # Add translation vector to current position (= get frame of goal position)
        root_T_tip_goal.p += root_V_hinge_drawer

        # Convert goal pose to dict for Giskard
        root_T_tip_goal_dict = convert_ros_message_to_dictionary(
            tf.kdl_to_pose_stamped(root_T_tip_goal, self.root))

        self.constraints.append(
            CartesianPoseStraight(
                god_map,
                self.root,
                self.tip,
                root_T_tip_goal_dict,
                weight=weight))

    def make_constraints(self):
        # Execute constraints
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())

    def __str__(self):
        s = super(OpenDrawer, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)


class Open(Constraint):
    def __init__(self, god_map, tip_link, object_name, object_link_name, root_link=None, goal_joint_state=None,
                 weight=WEIGHT_ABOVE_CA):
        super(Open, self).__init__(god_map)
        self.constraints = []
        environment_object = self.get_world().get_object(object_name)
        joint_name = environment_object.get_movable_parent_joint(object_link_name)

        if environment_object.is_joint_revolute(joint_name) or environment_object.is_joint_prismatic(joint_name):
            min_limit, max_limit = environment_object.get_joint_limits(joint_name)
            if goal_joint_state:
                goal_joint_state = min(max_limit, goal_joint_state)
            else:
                goal_joint_state = max_limit

        if environment_object.is_joint_revolute(joint_name):
            self.constraints.append(OpenDoor(god_map=god_map,
                                             tip_link=tip_link,
                                             object_name=object_name,
                                             object_link_name=object_link_name,
                                             angle_goal=goal_joint_state,
                                             root_link=root_link,
                                             weight=weight))
        elif environment_object.is_joint_prismatic(joint_name):
            self.constraints.append(OpenDrawer(god_map,
                                               tip_link=tip_link,
                                               object_name=object_name,
                                               object_link_name=object_link_name,
                                               distance_goal=goal_joint_state,
                                               root_link=root_link,
                                               weight=weight))
        else:
            logwarn(u'Opening containers with joint of type "{}" not supported'.format(
                environment_object.get_joint_type(joint_name)))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())


class Close(Constraint):
    def __init__(self, god_map, tip_link, object_name, object_link_name, root_link=None, goal_joint_state=None,
                 weight=WEIGHT_ABOVE_CA):
        super(Close, self).__init__(god_map)
        self.constraints = []
        environment_object = self.get_world().get_object(object_name)
        joint_name = environment_object.get_movable_parent_joint(object_link_name)

        if environment_object.is_joint_revolute(joint_name) or environment_object.is_joint_prismatic(joint_name):
            min_limit, max_limit = environment_object.get_joint_limits(joint_name)
            if goal_joint_state:
                goal_joint_state = max(min_limit, goal_joint_state)
            else:
                goal_joint_state = min_limit

        if environment_object.is_joint_revolute(joint_name):
            self.constraints.append(OpenDoor(god_map=god_map,
                                             tip_link=tip_link,
                                             object_name=object_name,
                                             object_link_name=object_link_name,
                                             angle_goal=goal_joint_state,
                                             root_link=root_link,
                                             weight=weight))
        elif environment_object.is_joint_prismatic(joint_name):
            self.constraints.append(OpenDrawer(god_map,
                                               tip_link=tip_link,
                                               object_name=object_name,
                                               object_link_name=object_link_name,
                                               distance_goal=goal_joint_state,
                                               root_link=root_link,
                                               weight=weight))
        else:
            logwarn(u'Opening containers with joint of type "{}" not supported'.format(
                environment_object.get_joint_type(joint_name)))

    def make_constraints(self):
        for constraint in self.constraints:
            self.soft_constraints.update(constraint.get_constraints())
