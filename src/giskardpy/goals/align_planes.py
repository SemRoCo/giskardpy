from typing import Optional

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import ColorRGBA, PrefixName
from giskardpy.goals.goal import Goal
from giskardpy.middleware import logging
from giskardpy.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.god_map import god_map


class AlignPlanes(Goal):
    def __init__(self,
                 root_link: PrefixName,
                 tip_link: PrefixName,
                 goal_normal: cas.Vector3,
                 tip_normal: cas.Vector3,
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
        :param reference_velocity: rad/s
        :param weight:
        """
        if 'root_normal' in kwargs:
            logging.logwarn('Deprecated warning: use goal_normal instead of root_normal')
            goal_normal = kwargs['root_normal']
        self.root = root_link
        self.tip = tip_link
        self.reference_velocity = reference_velocity
        self.weight = weight

        self.tip_V_tip_normal = god_map.world.transform(self.tip, tip_normal)
        self.tip_V_tip_normal.scale(1)

        self.root_V_root_normal = god_map.world.transform(self.root, goal_normal)
        self.root_V_root_normal.scale(1)

        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}' \
                   f'_X:{self.tip_V_tip_normal.x:.3f}' \
                   f'_Y:{self.tip_V_tip_normal.y:.3f}' \
                   f'_Z:{self.tip_V_tip_normal.z:.3f}'
        super().__init__(name)

        task = self.create_and_add_task('align planes')
        root_R_tip = god_map.world.compose_fk_expression(self.root, self.tip).to_rotation()
        root_V_tip_normal = root_R_tip.dot(self.tip_V_tip_normal)
        task.add_vector_goal_constraints(frame_V_current=root_V_tip_normal,
                                         frame_V_goal=self.root_V_root_normal,
                                         reference_velocity=self.reference_velocity,
                                         weight=self.weight)
        root_V_tip_normal.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_normal',
                                                              root_V_tip_normal,
                                                              color=ColorRGBA(r=1, g=0, b=0, a=1))
        self.root_V_root_normal.vis_frame = self.tip
        god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_normal',
                                                              self.root_V_root_normal,
                                                              color=ColorRGBA(r=0, g=0, b=1, a=1))
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
