from typing import Optional, List

from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import ColorRGBA

import giskardpy.utils.tfwrapper as tf
import giskardpy.casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.tasks.task import WEIGHT_ABOVE_CA, Task
from giskardpy.god_map import god_map
from giskardpy.utils.expression_definition_utils import transform_msg
from giskardpy.utils.logging import logwarn


class AlignPlanes(Goal):
    def __init__(self,
                 root_link: str,
                 tip_link: str,
                 goal_normal: Vector3Stamped,
                 tip_normal: Vector3Stamped,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 reference_velocity: float = 0.5,
                 weight: float = WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol,
                 **kwargs):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param goal_normal:
        :param tip_normal:
        :param root_group: if root_link is not unique, search in this group for matches.
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param reference_velocity: rad/s
        :param weight:
        """
        if 'root_normal' in kwargs:
            logwarn('Deprecated warning: use goal_normal instead of root_normal')
            goal_normal = kwargs['root_normal']
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        self.reference_velocity = reference_velocity
        self.weight = weight

        self.tip_V_tip_normal = transform_msg(self.tip, tip_normal)
        self.tip_V_tip_normal.vector = tf.normalize(self.tip_V_tip_normal.vector)

        self.root_V_root_normal = transform_msg(self.root, goal_normal)
        self.root_V_root_normal.vector = tf.normalize(self.root_V_root_normal.vector)

        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}' \
                   f'_X:{self.tip_V_tip_normal.vector.x:.3f}' \
                   f'_Y:{self.tip_V_tip_normal.vector.y:.3f}' \
                   f'_Z:{self.tip_V_tip_normal.vector.z:.3f}'
        super().__init__(name)

        task = self.create_and_add_task('align planes')
        tip_V_tip_normal = cas.Vector3(self.tip_V_tip_normal)
        root_R_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_rotation()
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        root_V_root_normal = cas.Vector3(self.root_V_root_normal)
        task.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=root_V_root_normal,
                                         reference_velocity=self.reference_velocity,
                                         weight=self.weight)
        root_V_tip_normal.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_normal',
                                                              root_V_tip_normal,
                                                              color=ColorRGBA(r=1, g=0, b=0, a=1))
        root_V_root_normal.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_normal',
                                                              root_V_root_normal,
                                                              color=ColorRGBA(r=0, g=0, b=1, a=1))
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
