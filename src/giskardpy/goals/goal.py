from __future__ import division

import numbers
from collections import OrderedDict

import numpy as np
from geometry_msgs.msg import Vector3Stamped, PoseStamped, PointStamped
from giskard_msgs.msg import Constraint as Constraint_msg

import giskardpy.identifier as identifier
from giskardpy import casadi_wrapper as w
from giskardpy.qp.constraint import VelocityConstraint, Constraint

WEIGHT_MAX = Constraint_msg.WEIGHT_MAX
WEIGHT_ABOVE_CA = 2500  # Constraint_msg.WEIGHT_ABOVE_CA
WEIGHT_COLLISION_AVOIDANCE = 50  # Constraint_msg.WEIGHT_COLLISION_AVOIDANCE
WEIGHT_BELOW_CA = 1  # Constraint_msg.WEIGHT_BELOW_CA
WEIGHT_MIN = Constraint_msg.WEIGHT_MIN


class Goal(object):
    def __init__(self, god_map, control_horizon=None, **kwargs):
        self.god_map = god_map
        self.prediction_horizon = self.get_god_map().get_data(identifier.prediction_horizon)
        # last 2 velocities are 0 anyway
        if control_horizon is None:
            control_horizon = self.prediction_horizon
        self.control_horizon = max(min(control_horizon, self.prediction_horizon - 2), 1)
        self.save_self_on_god_map()

    def save_self_on_god_map(self):
        self.get_god_map().set_data(self.get_identifier(), self)

    def make_constraints(self):
        pass

    def get_identifier(self):
        try:
            return identifier.goals + [str(self)]
        except AttributeError as e:
            raise AttributeError(
                u'You have to ensure that str(self) is possible before calling parents __init__: {}'.format(e))

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

    def get_joint_position_symbol(self, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        if not self.get_robot().has_joint(joint_name):
            raise KeyError('Robot doesn\'t have joint named: {}'.format(joint_name))
        key = identifier.joint_states + [joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_input_joint_velocity(self, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        key = identifier.joint_states + [joint_name, u'velocity']
        return self.god_map.to_symbol(key)

    def get_input_object_joint_position(self, object_name, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        key = identifier.world + [u'get_object', (object_name,), u'joint_state', joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_input_sampling_period(self):
        return self.god_map.to_symbol(identifier.sample_period)

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
        return self.get_god_map().to_transformation_matrix(identifier.fk_np + [(root, tip)])

    def get_parameter_as_symbolic_expression(self, name):
        """
        Returns a symbols that references a class attribute.
        :type name: str
        :return: w.Symbol
        """
        if not hasattr(self, name):
            raise AttributeError(u'{} doesn\'t have attribute {}'.format(self.__class__.__name__, name))
        key = self.get_identifier() + [name]
        value = self.get_god_map().get_data(key)
        if isinstance(value, numbers.Number):
            return self.get_input_float(name)
        elif isinstance(value, PoseStamped):
            return self.get_input_PoseStamped(name)
        elif isinstance(value, PointStamped):
            return self.get_input_PointStamped(name)
        elif isinstance(value, Vector3Stamped):
            return self.get_input_Vector3Stamped(name)
        elif isinstance(value, np.ndarray):
            return self.get_input_np_frame(name)
        else:
            raise NotImplementedError(u'Symbol reference not implemented for this type.')

    def get_input_float(self, name):
        """
        Returns a symbol that refers to the value of "name" on god map
        :type name: Union[str, unicode]
        :return: symbol
        """
        key = self.get_identifier() + [name]
        return self.god_map.to_symbol(key)

    def get_input_PoseStamped(self, name):
        """
        :param name: name of the god map entry
        :return: a homogeneous transformation matrix, with symbols that refer to a pose stamped in the god map.
        """
        return self.get_god_map().to_expr(self.get_identifier() + [name, u'pose'])

    def get_input_Vector3Stamped(self, name):
        return self.get_god_map().to_expr(self.get_identifier() + [name, u'vector'])

    def get_input_PointStamped(self, name):
        return self.get_god_map().to_expr(self.get_identifier() + [name, u'point'])

    def get_input_np_frame(self, name):
        return self.get_god_map().to_transformation_matrix(self.get_identifier() + [name])

    def get_expr_velocity(self, expr):
        return w.total_derivative(expr,
                                  self.get_robot().get_joint_position_symbols(),
                                  self.get_robot().get_joint_velocity_symbols())

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

    def limit_velocity(self, error, max_velocity):
        """
        :param error: expression that describes the error
        :param max_velocity: float or expression representing the max velocity
        :return: expression that limits the velocity of error to max_velocity
        """
        sample_period = self.get_input_sampling_period()
        max_velocity *= sample_period * self.control_horizon
        return w.max(w.min(error, max_velocity), -max_velocity)

    def normalize_weight(self, velocity_limit, weight):
        return weight

    def get_constraints(self):
        """
        :rtype: OrderedDict
        """
        self._constraints = OrderedDict()
        self._velocity_constraints = OrderedDict()
        self.debug_expressions = OrderedDict()
        self.make_constraints()
        return self._constraints, self._velocity_constraints

    def add_velocity_constraint(self, name_suffix, velocity_limit, weight, expression,
                                lower_slack_limit=-1e4, upper_slack_limit=1e4):

        name = str(self) + name_suffix
        if name in self._velocity_constraints:
            raise KeyError(u'a constraint with name \'{}\' already exists'.format(name))
        self._velocity_constraints[name] = VelocityConstraint(name=name,
                                                              expression=expression,
                                                              lower_velocity_limit=-velocity_limit,
                                                              upper_velocity_limit=velocity_limit,
                                                              quadratic_weight=weight,
                                                              lower_slack_limit=lower_slack_limit,
                                                              upper_slack_limit=upper_slack_limit,
                                                              control_horizon=self.control_horizon)

    def add_constraint(self, name_suffix, reference_velocity, lower_error,
                       upper_error, weight, expression, lower_slack_limit=-1e4, upper_slack_limit=1e4):

        name = str(self) + name_suffix
        if name in self._constraints:
            raise KeyError(u'a constraint with name \'{}\' already exists'.format(name))
        self._constraints[name] = Constraint(name=name,
                                             expression=expression,
                                             lower_error=lower_error,
                                             upper_error=upper_error,
                                             velocity_limit=reference_velocity,
                                             quadratic_weight=weight,
                                             lower_slack_limit=lower_slack_limit,
                                             upper_slack_limit=upper_slack_limit,
                                             control_horizon=self.control_horizon)

    def add_debug_expr(self, name, expr):
        """
        Adds a constraint with weight 0 to the qp problem.
        Used to inspect subexpressions for debugging.
        :param name: a name to identify the expression
        :type name: str
        :type expr: w.Symbol
        """
        name = str(self) + '/' + name
        self.debug_expressions[name] = expr

    def add_debug_matrix(self, name, matrix_expr):
        for x in range(matrix_expr.shape[0]):
            for y in range(matrix_expr.shape[1]):
                self.add_debug_expr(name + u'/{},{}'.format(x, y), matrix_expr[x, y])

    def add_debug_vector(self, name, vector_expr):
        for x in range(vector_expr.shape[0]):
            self.add_debug_expr(name + u'/{}'.format(x), vector_expr[x])

    def add_minimize_position_constraints(self, r_P_g, reference_velocity, root, tip, max_velocity=None,
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
        trans_error = w.norm(r_P_c)

        # trans_scale = w.scale(r_P_error, max_velocity)
        # r_P_intermediate_error = w.save_division(r_P_error, trans_error) * trans_scale

        weight = self.normalize_weight(max_velocity, weight)

        # self.add_debug_vector(u'r_P_error', r_P_error)
        # self.add_debug_vector(u'r_P_error/vel', self.get_expr_velocity(r_P_error))
        # self.add_debug_vector(u'trans_error', trans_error)
        # self.add_debug_vector(u'trans_error/v', w.norm(self.get_expr_velocity(r_P_error)))
        # self.add_debug_vector(u'trans_error/vel', self.get_expr_velocity(trans_error))

        self.add_constraint(u'{}/x'.format(prefix),
                            reference_velocity=reference_velocity,
                            lower_error=r_P_error[0],
                            upper_error=r_P_error[0],
                            weight=weight,
                            expression=r_P_c[0])
        self.add_constraint(u'{}/y'.format(prefix),
                            reference_velocity=reference_velocity,
                            lower_error=r_P_error[1],
                            upper_error=r_P_error[1],
                            weight=weight,
                            expression=r_P_c[1])
        self.add_constraint(u'{}/z'.format(prefix),
                            reference_velocity=reference_velocity,
                            lower_error=r_P_error[2],
                            upper_error=r_P_error[2],
                            weight=weight,
                            expression=r_P_c[2])
        if max_velocity is not None:
            self.add_velocity_constraint(u'{}/vel'.format(prefix),
                                         velocity_limit=max_velocity,
                                         weight=weight,
                                         expression=trans_error)

    def add_minimize_vector_angle_constraints(self, max_velocity, root, tip, tip_V_tip_normal, root_V_goal_normal,
                                              weight=WEIGHT_BELOW_CA, goal_constraint=False, prefix=u''):
        root_R_tip = w.rotation_of(self.get_fk(root, tip))
        root_V_tip_normal = w.dot(root_R_tip, tip_V_tip_normal)

        angle = w.save_acos(w.dot(root_V_tip_normal.T, root_V_goal_normal)[0])
        angle_limited = w.save_division(self.limit_velocity(angle, np.pi * 0.9),
                                        angle)  # avoid singularity by staying away from pi
        root_V_goal_normal_intermediate = w.slerp(root_V_tip_normal, root_V_goal_normal, angle_limited)
        error = root_V_goal_normal_intermediate - root_V_tip_normal

        weight = self.normalize_weight(max_velocity, weight)

        self.add_constraint(u'/{}/rot/x'.format(prefix),
                            reference_velocity=max_velocity,
                            lower_error=error[0],
                            upper_error=error[0],
                            weight=weight,
                            expression=root_V_tip_normal[0])
        self.add_constraint(u'/{}/rot/y'.format(prefix),
                            reference_velocity=max_velocity,
                            lower_error=error[1],
                            upper_error=error[1],
                            weight=weight,
                            expression=root_V_tip_normal[1])
        self.add_constraint(u'/{}/rot/z'.format(prefix),
                            reference_velocity=max_velocity,
                            lower_error=error[2],
                            upper_error=error[2],
                            weight=weight,
                            expression=root_V_tip_normal[2])

    def add_minimize_rotation_constraints(self, root_R_tipGoal, root, tip, reference_velocity, max_velocity,
                                          weight=WEIGHT_BELOW_CA, prefix=u''):
        root_R_tipCurrent = w.rotation_of(self.get_fk(root, tip))
        tip_R_rootCurrent_eval = w.rotation_of(self.get_fk_evaluated(tip, root))
        hack = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
        root_R_tipCurrent = w.dot(root_R_tipCurrent, hack)  # hack to avoid singularity
        tip_Q_tipCurrent = w.quaternion_from_matrix(w.dot(tip_R_rootCurrent_eval, root_R_tipCurrent))
        root_Q_tipCurrent = w.quaternion_from_matrix(root_R_tipCurrent)
        tip_R_goal = w.dot(tip_R_rootCurrent_eval, root_R_tipGoal)

        weight = self.normalize_weight(reference_velocity, weight)

        tip_Q_goal = w.quaternion_from_matrix(tip_R_goal)

        tip_Q_goal = w.if_greater_zero(-tip_Q_goal[3], -tip_Q_goal, tip_Q_goal)  # flip to get shortest path
        # angle_error = w.quaternion_angle(tip_Q_goal)
        angle_error2 = w.quaternion_angle(root_Q_tipCurrent)
        self.add_debug_expr('angle_error', angle_error2)
        # self.add_debug_vector('tip_Q_goal', tip_Q_goal)

        expr = tip_Q_tipCurrent

        self.add_constraint(u'{}/q/x'.format(prefix),
                            reference_velocity=reference_velocity,
                            lower_error=tip_Q_goal[0],
                            upper_error=tip_Q_goal[0],
                            weight=weight,
                            expression=expr[0])
        self.add_constraint(u'{}/q/y'.format(prefix),
                            reference_velocity=reference_velocity,
                            lower_error=tip_Q_goal[1],
                            upper_error=tip_Q_goal[1],
                            weight=weight,
                            expression=expr[1])
        self.add_constraint(u'{}/q/z'.format(prefix),
                            reference_velocity=reference_velocity,
                            lower_error=tip_Q_goal[2],
                            upper_error=tip_Q_goal[2],
                            weight=weight,
                            expression=expr[2])
        if max_velocity is not None:
            self.add_velocity_constraint(u'{}/q/vel'.format(prefix),
                                         velocity_limit=max_velocity,
                                         weight=weight * 1000,
                                         expression=angle_error2)
        # w is not needed because its derivative is always 0 for identity quaternions
