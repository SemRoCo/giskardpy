from __future__ import annotations
from typing import overload, TYPE_CHECKING

from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped

from giskardpy.god_map import god_map
import giskardpy.casadi_wrapper as cas

if TYPE_CHECKING:
    pass


@overload
def link_update_condition(data: cas.TransMatrix,
                          condition: cas.Expression) -> cas.TransMatrix: ...


@overload
def link_update_condition(data: cas.Point3,
                          condition: cas.Expression) -> cas.Point3: ...


@overload
def link_update_condition(data: cas.Vector3,
                          condition: cas.Expression) -> cas.Vector3: ...


@overload
def link_update_condition(data: cas.RotationMatrix,
                          condition: cas.Expression) -> cas.RotationMatrix: ...


def link_update_condition(data, condition):
    if not cas.is_true(condition):
        goal_frame_id = god_map.world.search_for_link_name(data.header.frame_id)
        root_T_goal_frame_id = god_map.world.compose_fk_expression(root_link, goal_frame_id)
        root_T_goal_frame_id = god_map.monitor_manager.register_expression_updater(root_T_goal_frame_id,
                                                                                   condition)
        if isinstance(data, PoseStamped):
            goal_frame_id_X_goal = cas.TransMatrix(goal_frame_id_X_goal)
        elif isinstance(data, PointStamped):
            goal_frame_id_X_goal = cas.Point3(goal_frame_id_X_goal)
        elif isinstance(data, Vector3Stamped):
            goal_frame_id_X_goal = cas.Vector3(goal_frame_id_X_goal)
        elif isinstance(data, QuaternionStamped):
            goal_frame_id_X_goal = cas.RotationMatrix(goal_frame_id_X_goal)
        root_X_goal = root_T_goal_frame_id.dot(goal_frame_id_X_goal)
        return root_X_goal
    else:
        transformed_msg = transform_msg(root_link, data)
        if isinstance(data, PoseStamped):
            return cas.TransMatrix(transformed_msg)
        if isinstance(data, PointStamped):
            return cas.Point3(transformed_msg)
        if isinstance(data, Vector3Stamped):
            return cas.Vector3(transformed_msg)
        if isinstance(data, QuaternionStamped):
            return cas.RotationMatrix(transformed_msg)
