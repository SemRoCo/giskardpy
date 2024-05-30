from typing import List, Optional, Union

from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped, Vector3Stamped

import giskardpy.casadi_wrapper as cas
from giskardpy.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.god_map import god_map
import giskardpy.utils.tfwrapper as tf
from giskardpy.utils.expression_definition_utils import transform_msg, transform_msg_and_turn_to_expr


class FeatureMonitor(ExpressionMonitor):
    def __init__(self,
                 tip_link: str, root_link: str,
                 world_feature: Union[PointStamped, Vector3Stamped],
                 robot_feature: Union[PointStamped, Vector3Stamped],
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
        if type(robot_feature) == PointStamped:
            self.root_P_robot_feature = root_T_tip.dot(cas.Point3(tip_robot_feature))
        elif type(robot_feature) == Vector3Stamped:
            self.root_V_robot_feature = root_T_tip.dot(cas.Vector3(tip_robot_feature))

        if type(world_feature) == PointStamped:
            self.root_P_world_feature = cas.Point3(root_world_feature)
        if type(world_feature) == Vector3Stamped:
            self.root_V_world_feature = cas.Vector3(root_world_feature)


class HeightFeatureMonitor(FeatureMonitor):
    def __init__(self,
                 tip_link: str, root_link: str,
                 world_feature: PointStamped,
                 robot_feature: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: Optional[str] = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name, stay_true=stay_true, start_condition=start_condition)

        distance = self.root_P_robot_feature - self.root_P_world_feature

        height_vector = cas.Vector3([0, 0, 1])

        projection = cas.dot(distance, height_vector)
        expr = cas.logic_and(cas.greater(projection, lower_limit), cas.less(projection, upper_limit))
        self.expression = expr


class PerpendicularFeatureMonitor(FeatureMonitor):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: Vector3Stamped,
                 robot_feature: Vector3Stamped,
                 threshold: float = 0.01,
                 name: str = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name, stay_true=stay_true, start_condition=start_condition)

        expr = cas.dot(self.root_V_world_feature[:3], self.root_V_robot_feature[:3])
        self.expression = cas.less_equal(cas.abs(expr), threshold)


class DistanceFeatureMonitor(FeatureMonitor):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: PointStamped,
                 robot_feature: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: str = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name, stay_true=stay_true, start_condition=start_condition)

        distance_vector = self.root_P_robot_feature - self.root_P_world_feature
        height_vector = cas.Vector3([0, 0, 1])
        projection = cas.norm(distance_vector - cas.dot(distance_vector, height_vector) * height_vector)
        self.expression = cas.logic_and(cas.greater(projection, lower_limit), cas.less(projection, upper_limit))


#TODO: PointingAt monitor and VectorAligned monitor already exist

class AngleFeatureMonitor(FeatureMonitor):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: Vector3Stamped,
                 robot_feature: Vector3Stamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: str = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name, stay_true=stay_true, start_condition=start_condition)

        expr = cas.angle_between_vector(self.root_V_world_feature, self.root_V_robot_feature)
        self.expression = cas.logic_and(cas.greater(expr, lower_limit), cas.less(expr, upper_limit))
