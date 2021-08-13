from __future__ import division

from collections import OrderedDict

import numpy as np
from giskard_msgs.msg import Constraint as Constraint_msg

import giskardpy.identifier as identifier
from giskardpy import casadi_wrapper as w
from giskardpy.exceptions import ConstraintInitalizationException
from giskardpy.model.robot import Robot
from giskardpy.qp.constraint import VelocityConstraint, Constraint

WEIGHT_MAX = Constraint_msg.WEIGHT_MAX
WEIGHT_ABOVE_CA = 2500  # Constraint_msg.WEIGHT_ABOVE_CA
WEIGHT_COLLISION_AVOIDANCE = 50  # Constraint_msg.WEIGHT_COLLISION_AVOIDANCE
WEIGHT_BELOW_CA = 1  # Constraint_msg.WEIGHT_BELOW_CA
WEIGHT_MIN = Constraint_msg.WEIGHT_MIN


class Goal(object):
    def __init__(self, god_map, control_horizon=None, **kwargs):
        self._god_map = god_map
        self.prediction_horizon = self.get_god_map().get_data(identifier.prediction_horizon)
        self._test_mode = self.get_god_map().get_data(identifier.test_mode)
        # last 2 velocities are 0 anyway
        if control_horizon is None:
            control_horizon = self.prediction_horizon
        self.control_horizon = max(min(control_horizon, self.prediction_horizon - 2), 1)
        self._sub_goals = []

    def _save_self_on_god_map(self):
        self.get_god_map().set_data(self._get_identifier(), self)

    def make_constraints(self):
        pass

    def _get_identifier(self):
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
        return self._god_map

    def get_world(self):
        """
        :rtype: giskardpy.world.World
        """
        return self.get_god_map().get_data(identifier.world)

    def get_robot(self):
        """
        :rtype: Robot
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
        return self._god_map.to_symbol(key)

    def get_joint_velocity_symbols(self, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        key = identifier.joint_states + [joint_name, u'velocity']
        return self._god_map.to_symbol(key)

    def get_object_joint_position_symbol(self, object_name, joint_name):
        """
        returns a symbol that referes to the given joint
        """
        # TODO test me
        key = identifier.world + [u'get_object', (object_name,), u'joint_state', joint_name, u'position']
        return self._god_map.to_symbol(key)

    def get_sampling_period_symbol(self):
        return self._god_map.to_symbol(identifier.sample_period)

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
        return self.get_god_map().list_to_frame(identifier.fk_np + [(root, tip)])

    def get_parameter_as_symbolic_expression(self, name):
        """
        Returns a symbols that references a class attribute.
        :type name: str
        :return: w.Symbol
        """
        if not hasattr(self, name):
            raise AttributeError(u'{} doesn\'t have attribute {}'.format(self.__class__.__name__, name))
        return self.get_god_map().to_expr(self._get_identifier() + [name])

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

    def get_constraints(self):
        """
        :rtype: OrderedDict
        """
        self._save_self_on_god_map()
        self._constraints = OrderedDict()
        self._velocity_constraints = OrderedDict()
        self._debug_expressions = OrderedDict()
        self.make_constraints()
        for sub_goal in self._sub_goals:
            c, c_vel, debug_expressions = sub_goal.get_constraints()
            # TODO check for duplicates
            self._constraints.update(_prepend_prefix(self.__class__.__name__, c))
            self._velocity_constraints.update(_prepend_prefix(self.__class__.__name__, c_vel))
            self._debug_expressions.update(_prepend_prefix(self.__class__.__name__, debug_expressions))
        return self._constraints, self._velocity_constraints, self._debug_expressions

    def add_constraints_of_goal(self, goal):
        self._sub_goals.append(goal)

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

    def add_constraint(self, reference_velocity, lower_error, upper_error, weight, expression, name_suffix=None,
                       lower_slack_limit=None, upper_slack_limit=None):

        name_suffix = name_suffix if name_suffix else u''
        name = str(self) + name_suffix
        if name in self._constraints:
            raise KeyError(u'a constraint with name \'{}\' already exists'.format(name))
        lower_slack_limit = lower_slack_limit if lower_slack_limit is not None else -1e4
        upper_slack_limit = upper_slack_limit if upper_slack_limit is not None else 1e4
        self._constraints[name] = Constraint(name=name,
                                             expression=expression,
                                             lower_error=lower_error,
                                             upper_error=upper_error,
                                             velocity_limit=reference_velocity,
                                             quadratic_weight=weight,
                                             lower_slack_limit=lower_slack_limit,
                                             upper_slack_limit=upper_slack_limit,
                                             control_horizon=self.control_horizon)

    def add_constraint_vector(self, reference_velocities, lower_errors, upper_errors, weights, expressions,
                              name_suffixes=None, lower_slack_limits=None, upper_slack_limits=None):
        reference_velocities = w.matrix_to_list(reference_velocities)
        lower_errors = w.matrix_to_list(lower_errors)
        upper_errors = w.matrix_to_list(upper_errors)
        weights = w.matrix_to_list(weights)
        expressions = w.matrix_to_list(expressions)
        if lower_slack_limits is not None:
            lower_slack_limits = w.matrix_to_list(lower_slack_limits)
        if upper_slack_limits is not None:
            upper_slack_limits = w.matrix_to_list(upper_slack_limits)
        if len(lower_errors) != len(upper_errors) \
                or len(lower_errors) != len(expressions) \
                or len(lower_errors) != len(reference_velocities) \
                or len(lower_errors) != len(weights) \
                or (name_suffixes is not None and len(lower_errors) != len(name_suffixes)) \
                or (lower_slack_limits is not None and len(lower_errors) != len(lower_slack_limits)) \
                or (upper_slack_limits is not None and len(lower_errors) != len(upper_slack_limits)):
            raise ConstraintInitalizationException('All parameters must have the same length.')
        for i in range(len(lower_errors)):
            name_suffix = name_suffixes[i] if name_suffixes else None
            lower_slack_limit = lower_slack_limits[i] if lower_slack_limits else None
            upper_slack_limit = upper_slack_limits[i] if upper_slack_limits else None
            self.add_constraint(reference_velocity=reference_velocities[i],
                                lower_error=lower_errors[i],
                                upper_error=upper_errors[i],
                                weight=weights[i],
                                expression=expressions[i],
                                name_suffix=name_suffix,
                                lower_slack_limit=lower_slack_limit,
                                upper_slack_limit=upper_slack_limit)

    def add_debug_expr(self, name, expr):
        """
        Adds a constraint with weight 0 to the qp problem.
        Used to inspect subexpressions for debugging.
        :param name: a name to identify the expression
        :type name: str
        :type expr: w.Symbol
        """
        name = str(self) + '/' + name
        self._debug_expressions[name] = expr

    def add_debug_matrix(self, name, matrix_expr):
        for x in range(matrix_expr.shape[0]):
            for y in range(matrix_expr.shape[1]):
                self.add_debug_expr(name + u'/{},{}'.format(x, y), matrix_expr[x, y])

    def add_debug_vector(self, name, vector_expr):
        for x in range(vector_expr.shape[0]):
            self.add_debug_expr(name + u'/{}'.format(x), vector_expr[x])

    def add_position_constraint(self, expr_current, expr_goal, reference_velocity, weight=WEIGHT_BELOW_CA,
                                name_suffix=u''):

        error = expr_goal - expr_current
        self.add_constraint(reference_velocity=reference_velocity,
                            lower_error=error,
                            upper_error=error,
                            weight=weight,
                            expression=expr_current,
                            name_suffix=name_suffix)

    def add_point_goal_constraints(self, frame_P_current, frame_P_goal, reference_velocity, weight, name_suffix=u''):
        error = frame_P_goal[:3] - frame_P_current[:3]
        self.add_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                   lower_errors=error[:3],
                                   upper_errors=error[:3],
                                   weights=[weight] * 3,
                                   expressions=frame_P_current[:3],
                                   name_suffixes=['{}/x'.format(name_suffix),
                                                  '{}/y'.format(name_suffix),
                                                  '{}/z'.format(name_suffix)])
        if self._test_mode:
            self.add_debug_expr('{}/error'.format(name_suffix), w.norm(error))

    def add_translational_velocity_limit(self, frame_P_current, max_velocity, weight, max_violation=1e4, name_suffix=u''):
        trans_error = w.norm(frame_P_current)
        self.add_velocity_constraint(velocity_limit=max_velocity,
                                     weight=weight,
                                     expression=trans_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     name_suffix=u'{}/vel'.format(name_suffix))
        if self._test_mode:
            self.add_debug_expr('trans_error', self.get_expr_velocity(trans_error))

    def add_vector_goal_constraints(self, frame_V_current, frame_V_goal, reference_velocity,
                                    weight=WEIGHT_BELOW_CA, name_suffix=u''):

        angle = w.save_acos(w.dot(frame_V_current.T, frame_V_goal)[0])
        # avoid singularity by staying away from pi
        angle_limited = w.min(w.max(angle, -reference_velocity), reference_velocity)
        angle_limited = w.save_division(angle_limited, angle)
        root_V_goal_normal_intermediate = w.slerp(frame_V_current, frame_V_goal, angle_limited)

        error = root_V_goal_normal_intermediate - frame_V_current

        self.add_constraint_vector(reference_velocities=[reference_velocity]*3,
                                   lower_errors=error[:3],
                                   upper_errors=error[:3],
                                   weights=[weight]*3,
                                   expressions=frame_V_current[:3],
                                   name_suffixes=[u'{}/trans/x'.format(name_suffix),
                                                  u'{}/trans/y'.format(name_suffix),
                                                  u'{}/trans/z'.format(name_suffix)])

    def add_rotation_goal_constraints(self, frame_R_current, frame_R_goal, current_R_frame_eval, reference_velocity,
                                      weight, name_suffix=u''):
        hack = w.rotation_matrix_from_axis_angle([0, 0, 1], 0.0001)
        frame_R_current = w.dot(frame_R_current, hack)  # hack to avoid singularity
        tip_Q_tipCurrent = w.quaternion_from_matrix(w.dot(current_R_frame_eval, frame_R_current))
        tip_R_goal = w.dot(current_R_frame_eval, frame_R_goal)

        tip_Q_goal = w.quaternion_from_matrix(tip_R_goal)

        tip_Q_goal = w.if_greater_zero(-tip_Q_goal[3], -tip_Q_goal, tip_Q_goal)  # flip to get shortest path

        expr = tip_Q_tipCurrent
        # w is not needed because its derivative is always 0 for identity quaternions
        self.add_constraint_vector(reference_velocities=[reference_velocity] * 3,
                                   lower_errors=tip_Q_goal[:3],
                                   upper_errors=tip_Q_goal[:3],
                                   weights=[weight] * 3,
                                   expressions=expr[:3],
                                   name_suffixes=['{}/rot/x'.format(name_suffix),
                                                  '{}/rot/y'.format(name_suffix),
                                                  '{}/rot/z'.format(name_suffix)])

    def add_rotational_velocity_limit(self, frame_R_current, max_velocity, weight, max_violation=1e4, name_suffix=u''):
        root_Q_tipCurrent = w.quaternion_from_matrix(frame_R_current)
        angle_error = w.quaternion_angle(root_Q_tipCurrent)
        self.add_velocity_constraint(velocity_limit=max_velocity,
                                     weight=weight,
                                     expression=angle_error,
                                     lower_slack_limit=-max_violation,
                                     upper_slack_limit=max_violation,
                                     name_suffix=u'{}/q/vel'.format(name_suffix))

def _prepend_prefix(prefix, d):
    new_dict = OrderedDict()
    for key, value in d.items():
        new_key = u'{}/{}'.format(prefix, key)
        value.name = u'{}/{}'.format(prefix, value.name)
        new_dict[new_key] = value
    return new_dict