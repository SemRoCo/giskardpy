from __future__ import annotations
from typing import overload, TYPE_CHECKING

from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, QuaternionStamped

from giskardpy.exceptions import UnknownGroupException
from giskardpy.god_map import god_map
from giskardpy.data_types import my_string, PrefixName
import giskardpy.utils.tfwrapper as tf
import giskardpy.casadi_wrapper as cas

if TYPE_CHECKING:
    pass


@overload
def transform_msg(target_frame: my_string, msg: PoseStamped, tf_timeout: float = 1) -> PoseStamped: ...


@overload
def transform_msg(target_frame: my_string, msg: PointStamped, tf_timeout: float = 1) -> PointStamped: ...


@overload
def transform_msg(target_frame: my_string, msg: Vector3Stamped, tf_timeout: float = 1) -> Vector3Stamped: ...


@overload
def transform_msg(target_frame: my_string, msg: QuaternionStamped, tf_timeout: float = 1) -> QuaternionStamped: ...


def transform_msg(target_frame, msg, tf_timeout=1):
    """
    First tries to transform the message using the world's internal kinematic tree.
    If it fails, it uses tf as a backup.
    :param target_frame:
    :param msg:
    :param tf_timeout: for how long Giskard should wait for tf.
    :return: message relative to target frame
    """
    try:
        try:
            msg.header.frame_id = god_map.world.search_for_link_name(msg.header.frame_id)
        except UnknownGroupException:
            pass
        return god_map.world.transform_msg(target_frame, msg)
    except KeyError:
        return tf.transform_msg(target_frame, msg, timeout=tf_timeout)


@overload
def transform_msg_and_turn_to_expr(root_link: PrefixName,
                                   msg: PoseStamped,
                                   condition: cas.Expression) -> cas.TransMatrix: ...


@overload
def transform_msg_and_turn_to_expr(root_link: PrefixName,
                                   msg: PointStamped,
                                   condition: cas.Expression) -> cas.Point3: ...


@overload
def transform_msg_and_turn_to_expr(root_link: PrefixName,
                                   msg: Vector3Stamped,
                                   condition: cas.Expression) -> cas.Vector3: ...


@overload
def transform_msg_and_turn_to_expr(root_link: PrefixName,
                                   msg: QuaternionStamped,
                                   condition: cas.Expression) -> cas.RotationMatrix: ...


def transform_msg_and_turn_to_expr(root_link, msg, condition):
    if not cas.is_true(condition):
        goal_frame_id = god_map.world.search_for_link_name(msg.header.frame_id)
        goal_frame_id_X_goal = transform_msg(goal_frame_id, msg)
        root_T_goal_frame_id = god_map.world.compose_fk_expression(root_link, goal_frame_id)
        root_T_goal_frame_id = god_map.monitor_manager.register_expression_updater(root_T_goal_frame_id,
                                                                                   condition)
        if isinstance(msg, PoseStamped):
            goal_frame_id_X_goal = cas.TransMatrix(goal_frame_id_X_goal)
        elif isinstance(msg, PointStamped):
            goal_frame_id_X_goal = cas.Point3(goal_frame_id_X_goal)
        elif isinstance(msg, Vector3Stamped):
            goal_frame_id_X_goal = cas.Vector3(goal_frame_id_X_goal)
        elif isinstance(msg, QuaternionStamped):
            goal_frame_id_X_goal = cas.RotationMatrix(goal_frame_id_X_goal)
        root_X_goal = root_T_goal_frame_id.dot(goal_frame_id_X_goal)
        return root_X_goal
    else:
        transformed_msg = transform_msg(root_link, msg)
        if isinstance(msg, PoseStamped):
            return cas.TransMatrix(transformed_msg)
        if isinstance(msg, PointStamped):
            return cas.Point3(transformed_msg)
        if isinstance(msg, Vector3Stamped):
            return cas.Vector3(transformed_msg)
        if isinstance(msg, QuaternionStamped):
            return cas.RotationMatrix(transformed_msg)
