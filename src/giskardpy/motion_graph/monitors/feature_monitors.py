from typing import List, Optional, Union

from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped, Vector3Stamped

import giskardpy.casadi_wrapper as cas
from giskardpy.motion_graph.monitors.monitors import ExpressionMonitor, Monitor
from giskardpy.god_map import god_map
import giskardpy.utils.tfwrapper as tf
from giskardpy.utils.expression_definition_utils import transform_msg, transform_msg_and_turn_to_expr


class FeatureMonitor(ExpressionMonitor):
    def __init__(self,
                 tip_link: str, root_link: str,
                 reference_feature: Union[PointStamped, Vector3Stamped],
                 controlled_feature: Union[PointStamped, Vector3Stamped],
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name, start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition)
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)

        reference_feature.header.frame_id = god_map.world.search_for_link_name(reference_feature.header.frame_id, None)
        root_reference_feature = god_map.world.transform_msg(self.root, reference_feature)
        controlled_feature.header.frame_id = god_map.world.search_for_link_name(controlled_feature.header.frame_id,
                                                                                None)
        tip_controlled_feature = god_map.world.transform_msg(self.tip, controlled_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        if isinstance(controlled_feature, PointStamped):
            self.root_P_controlled_feature = root_T_tip.dot(cas.Point3(tip_controlled_feature))
        elif isinstance(controlled_feature, Vector3Stamped):
            self.root_V_controlled_feature = root_T_tip.dot(cas.Vector3(tip_controlled_feature))

        if isinstance(reference_feature, PointStamped):
            self.root_P_reference_feature = cas.Point3(root_reference_feature)
        if isinstance(reference_feature, Vector3Stamped):
            self.root_V_reference_feature = cas.Vector3(root_reference_feature)


class HeightMonitor(FeatureMonitor):
    def __init__(self,
                 tip_link: str, root_link: str,
                 reference_point: PointStamped,
                 tip_point: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 name: Optional[str] = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_point,
                         controlled_feature=tip_point, name=name, start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         root_group=root_group, tip_group=tip_group)

        distance = cas.distance_projected_on_vector(self.root_P_controlled_feature, self.root_P_reference_feature,
                                                    cas.Vector3([0, 0, 1]))
        expr = cas.logic_and(cas.greater_equal(distance, lower_limit), cas.less_equal(distance, upper_limit))
        self.expression = expr


class PerpendicularMonitor(FeatureMonitor):
    def __init__(self, tip_link: str, root_link: str,
                 reference_normal: Vector3Stamped,
                 tip_normal: Vector3Stamped,
                 threshold: float = 0.01,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 name: str = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_normal,
                         controlled_feature=tip_normal, name=name, start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         root_group=root_group, tip_group=tip_group)

        expr = cas.dot(self.root_V_reference_feature[:3], self.root_V_controlled_feature[:3])
        self.expression = cas.less_equal(cas.abs(expr), threshold)


class DistanceMonitor(FeatureMonitor):
    def __init__(self, tip_link: str, root_link: str,
                 reference_point: PointStamped,
                 tip_point: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 name: str = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_point,
                         controlled_feature=tip_point, name=name, start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         root_group=root_group, tip_group=tip_group)

        distance = cas.norm(
            cas.distance_vector_projected_on_plane(self.root_P_controlled_feature, self.root_P_reference_feature,
                                                   cas.Vector3([0, 0, 1])))
        self.expression = cas.logic_and(cas.greater_equal(distance, lower_limit), cas.less_equal(distance, upper_limit))


class AngleMonitor(FeatureMonitor):
    def __init__(self, tip_link: str, root_link: str,
                 reference_vector: Vector3Stamped,
                 tip_vector: Vector3Stamped,
                 lower_angle: float,
                 upper_angle: float,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 name: str = None,
                 stay_true: bool = True,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_vector,
                         controlled_feature=tip_vector, name=name, start_condition=start_condition,
                         hold_condition=hold_condition,
                         end_condition=end_condition,
                         root_group=root_group, tip_group=tip_group)

        expr = cas.angle_between_vector(self.root_V_reference_feature, self.root_V_controlled_feature)
        self.expression = cas.logic_and(cas.greater(expr, lower_angle), cas.less(expr, upper_angle))
