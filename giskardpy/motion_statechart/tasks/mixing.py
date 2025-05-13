from typing import Optional

from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition, CartesianOrientation
from giskardpy.motion_statechart.tasks.task import Task, WEIGHT_BELOW_CA
import giskardpy.casadi_wrapper as cas
from giskardpy.symbol_manager import symbol_manager


class SpiralMixing(Task):
    def __init__(self, *, name: Optional[str] = None,
                 end_time: float,
                 object_name: PrefixName,
                 tool_height: float,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 radial_increment: float,
                 angle_increment: float,
                 upward_increment: float,
                 weight: float = WEIGHT_BELOW_CA,
                 plot: bool = True):

        super().__init__(name=name, plot=plot)

        root_T_tip = god_map.world.compose_fk_expression(root_link=root_link, tip_link=tip_link)
        t = symbol_manager.time

        r = radial_increment * t
        a = angle_increment * t
        h = upward_increment * t

        object_T_goal = cas.TransMatrix()
        x = r * cas.cos(a)
        y = r * cas.sin(a)
        z = h

        object_T_goal.x = x
        object_T_goal.y = y
        object_T_goal.z = z

        root_T_object = god_map.world.compose_fk_expression(root_link=root_link,
                                                            tip_link=object_name)
        root_T_goal = root_T_object.dot(object_T_goal)
        root_T_goal.z += tool_height + 0.05

        self.add_point_goal_constraints(frame_P_current=root_T_tip.to_position(),
                                        frame_P_goal=root_T_goal.to_position(),
                                        reference_velocity=CartesianPosition.default_reference_velocity,
                                        weight=weight)
        god_map.debug_expression_manager.add_debug_expression('root_T_goal', root_T_goal)
        self.add_rotation_goal_constraints(frame_R_current=root_T_tip.to_rotation(),
                                           frame_R_goal=root_T_goal.to_rotation(),
                                           current_R_frame_eval=god_map.world.compose_fk_evaluated_expression(root=tip_link,
                                                                                                              tip=root_link),
                                           reference_velocity=CartesianOrientation.default_reference_velocity,
                                           weight=weight)

        self.observation_expression = cas.greater(t, end_time)
