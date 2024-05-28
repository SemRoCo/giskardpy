from typing import List, Optional

from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped, Vector3Stamped

import giskardpy.casadi_wrapper as cas
from giskardpy.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.god_map import god_map
import giskardpy.utils.tfwrapper as tf
from giskardpy.utils.expression_definition_utils import transform_msg, transform_msg_and_turn_to_expr


class HeightFeature(ExpressionMonitor):
    def __init__(self,
                 tip_link: str, root_link: str,
                 world_feature: PointStamped,
                 robot_feature: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: Optional[str] = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(name=name, stay_true=stay_true, start_condition=start_condition)
        self.root = god_map.world.search_for_link_name(root_link, None)
        self.tip = god_map.world.search_for_link_name(tip_link, None)

        world_feature.header.frame_id = god_map.world.search_for_link_name(world_feature.header.frame_id, None)
        root_world_feature = god_map.world.transform_msg(self.root, world_feature)
        robot_feature.header.frame_id = god_map.world.search_for_link_name(robot_feature.header.frame_id, None)
        tip_robot_feature = god_map.world.transform_msg(self.tip, robot_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        root_P_robot_feature = root_T_tip.dot(cas.Point3(tip_robot_feature))
        root_P_world_feature = cas.Point3(root_world_feature)

        distance = root_P_robot_feature - root_P_world_feature

        height_vector = cas.Vector3([0, 0, 1])

        projection = cas.dot(distance, height_vector)
        expr = cas.logic_and(cas.greater(projection, lower_limit), cas.less(projection, upper_limit))
        self.expression = expr
