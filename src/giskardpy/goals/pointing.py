from __future__ import division

from geometry_msgs.msg import Vector3Stamped

import giskardpy.identifier as identifier
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA
import giskardpy.utils.tfwrapper as tf

class Pointing(Goal):

    def __init__(self, tip_link, goal_point, root_link, pointing_axis=None, max_velocity=0.3,
                 weight=WEIGHT_BELOW_CA, **kwargs):
        """
        Uses the kinematic chain from root_link to tip_link to move the pointing axis, such that it points to the goal point.
        :param tip_link: str, name of the tip of the kin chain
        :param goal_point: PointStamped as json, where the pointing_axis will point towards
        :param root_link: str, name of the root of the kin chain
        :param pointing_axis: Vector3Stamped as json, default is z axis, this axis will point towards the goal_point
        :param weight: float, default WEIGHT_BELOW_CA
        """
        super(Pointing, self).__init__(**kwargs)
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = root_link
        self.tip = tip_link
        self.root_P_goal_point = tf.transform_point(self.root, goal_point)

        if pointing_axis is not None:
            self.tip_V_pointing_axis = tf.transform_vector(self.tip, pointing_axis)
            self.tip_V_pointing_axis.vector = tf.normalize(self.tip_V_pointing_axis.vector)
        else:
            self.tip_V_pointing_axis = Vector3Stamped()
            self.tip_V_pointing_axis.header.frame_id = self.tip
            self.tip_V_pointing_axis.vector.z = 1


    def make_constraints(self):
        # TODO fix comments
        # in this function, you have to create the actual constraints
        # start by creating references to your input params in the god map
        # get_input functions generally return symbols referring to god map entries
        root_T_tip = self.get_fk(self.root, self.tip)
        root_P_goal_point = self.get_parameter_as_symbolic_expression(u'root_P_goal_point')
        tip_V_pointing_axis = self.get_parameter_as_symbolic_expression(u'tip_V_pointing_axis')

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
        root_V_goal_axis = root_P_goal_point - w.position_of(root_T_tip)
        root_V_goal_axis /= w.norm(root_V_goal_axis)  # FIXME avoid /0
        root_V_pointing_axis = w.dot(root_T_tip, tip_V_pointing_axis)

        # add constraints to the current problem, after execution, it gets cleared automatically
        self.add_vector_goal_constraints(frame_V_current=root_V_pointing_axis,
                                         frame_V_goal=root_V_goal_axis,
                                         reference_velocity=self.max_velocity,
                                         weight=self.weight)
        # self.add_constraint(
        #     u'x',
        #     reference_velocity=max_velocity,
        #     lower_error=diff[0],
        #     upper_error=diff[0],
        #     weight=weight,
        #     expression=current_axis[0])
        #
        # self.add_constraint(u'y',
        #                     reference_velocity=max_velocity,
        #                     lower_error=diff[1],
        #                     upper_error=diff[1],
        #                     weight=weight,
        #                     expression=current_axis[1])
        # self.add_constraint(u'z',
        #                     reference_velocity=max_velocity,
        #                     lower_error=diff[2],
        #                     upper_error=diff[2],
        #                     weight=weight,
        #                     expression=current_axis[2])

    def __str__(self):
        # helps to make sure your constraint name is unique.
        s = super(Pointing, self).__str__()
        return u'{}/{}/{}'.format(s, self.root, self.tip)
