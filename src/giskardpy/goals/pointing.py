from __future__ import division

from typing import Optional, List

from geometry_msgs.msg import Vector3Stamped, PointStamped
from std_msgs.msg import ColorRGBA

import giskardpy.utils.tfwrapper as tf
import giskardpy.casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.symbol_manager import symbol_manager
from giskardpy.tasks.task import WEIGHT_BELOW_CA, Task
from giskardpy.god_map import god_map
from giskardpy.utils.expression_definition_utils import transform_msg
from giskardpy.utils.logging import logwarn


class Pointing(Goal):
    def __init__(self,
                 tip_link: str,
                 goal_point: PointStamped,
                 root_link: str,
                 tip_group: Optional[str] = None,
                 root_group: Optional[str] = None,
                 pointing_axis: Vector3Stamped = None,
                 max_velocity: float = 0.3,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.root_P_goal_point = transform_msg(self.root, goal_point)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name)

        if pointing_axis is not None:
            self.tip_V_pointing_axis = transform_msg(self.tip, pointing_axis)
            self.tip_V_pointing_axis.vector = tf.normalize(self.tip_V_pointing_axis.vector)
        else:
            logwarn(f'Deprecated warning: Please set pointing_axis.')
            self.tip_V_pointing_axis = Vector3Stamped()
            self.tip_V_pointing_axis.header.frame_id = self.tip
            self.tip_V_pointing_axis.vector.z = 1

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_goal_point = symbol_manager.get_expr(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\']'
                                                    f'.root_P_goal_point',
                                                    input_type_hint=PointStamped,
                                                    output_type_hint=cas.Point3)
        root_P_goal_point.reference_frame = self.root
        tip_V_pointing_axis = cas.Vector3(self.tip_V_pointing_axis)

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip
        root_V_goal_axis.vis_frame = self.tip
        # self.add_debug_expr('goal_point', root_P_goal_point)
        # self.add_debug_expr('root_V_pointing_axis', root_V_pointing_axis)
        # self.add_debug_expr('root_V_goal_axis', root_V_goal_axis)
        god_map.debug_expression_manager.add_debug_expression('root_V_pointing_axis',
                                                              root_V_pointing_axis,
                                                              color=ColorRGBA(r=1, g=0, b=0, a=1))
        god_map.debug_expression_manager.add_debug_expression('goal_point',
                                                              root_P_goal_point,
                                                              color=ColorRGBA(r=0, g=0, b=1, a=1))
        task = self.create_and_add_task('pointing')
        task.add_vector_goal_constraints(frame_V_current=root_V_pointing_axis,
                                         frame_V_goal=root_V_goal_axis,
                                         reference_velocity=self.max_velocity,
                                         weight=self.weight)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
