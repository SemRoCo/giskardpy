from __future__ import division

from typing import Optional

import numpy as np

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import ColorRGBA, PrefixName
from giskardpy.goals.goal import Goal
from giskardpy.symbol_manager import symbol_manager
from giskardpy.tasks.task import WEIGHT_BELOW_CA
from giskardpy.god_map import god_map


class Pointing(Goal):
    def __init__(self,
                 tip_link: PrefixName,
                 goal_point: cas.Point3,
                 root_link: PrefixName,
                 pointing_axis: cas.Vector3,
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
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.root = root_link
        self.tip = tip_link
        self.root_P_goal_point = god_map.world.transform(self.root, goal_point).to_np()
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name)

        self.tip_V_pointing_axis = god_map.world.transform(self.tip, pointing_axis)
        self.tip_V_pointing_axis.scale(1)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_goal_point = symbol_manager.get_expr(f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\']'
                                                    f'.root_P_goal_point',
                                                    input_type_hint=np.ndarray,
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
