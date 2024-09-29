from typing import Optional, Union

import giskardpy.casadi_wrapper as cas
from giskardpy.data_types.data_types import PrefixName
from giskardpy.motion_graph.monitors.monitors import Monitor
from giskardpy.god_map import god_map


class FeatureMonitor(Monitor):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 reference_feature: Union[cas.Point3, cas.Vector3],
                 controlled_feature: Union[cas.Point3, cas.Vector3],
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(name=name, start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)
        self.root = root_link
        self.tip = tip_link

        root_reference_feature = god_map.world.transform(self.root, reference_feature)
        tip_controlled_feature = god_map.world.transform(self.tip, controlled_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        if isinstance(controlled_feature, cas.Point3):
            self.root_P_controlled_feature = root_T_tip.dot(tip_controlled_feature)
        elif isinstance(controlled_feature, cas.Vector3):
            self.root_V_controlled_feature = root_T_tip.dot(tip_controlled_feature)

        if isinstance(reference_feature, cas.Point3):
            self.root_P_reference_feature = root_reference_feature
        if isinstance(reference_feature, cas.Vector3):
            self.root_V_reference_feature = root_reference_feature


class HeightMonitor(FeatureMonitor):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 reference_point: cas.Point3,
                 tip_point: cas.Point3,
                 lower_limit: float,
                 upper_limit: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(tip_link=tip_link,
                         root_link=root_link,
                         reference_feature=reference_point,
                         controlled_feature=tip_point,
                         name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)

        distance = cas.distance_projected_on_vector(self.root_P_controlled_feature, self.root_P_reference_feature,
                                                    cas.Vector3([0, 0, 1]))
        expr = cas.logic_and(cas.greater_equal(distance, lower_limit), cas.less_equal(distance, upper_limit))
        self.expression = expr


class PerpendicularMonitor(FeatureMonitor):
    def __init__(self, tip_link: PrefixName,
                 root_link: PrefixName,
                 reference_normal: cas.Vector3,
                 tip_normal: cas.Vector3,
                 threshold: float = 0.01,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(tip_link=tip_link,
                         root_link=root_link,
                         reference_feature=reference_normal,
                         controlled_feature=tip_normal,
                         name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)

        expr = cas.dot(self.root_V_reference_feature[:3], self.root_V_controlled_feature[:3])
        self.expression = cas.less_equal(cas.abs(expr), threshold)


class DistanceMonitor(FeatureMonitor):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 reference_point: cas.Point3,
                 tip_point: cas.Point3,
                 lower_limit: float,
                 upper_limit: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(tip_link=tip_link,
                         root_link=root_link,
                         reference_feature=reference_point,
                         controlled_feature=tip_point,
                         name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)

        distance = cas.norm(cas.distance_vector_projected_on_plane(self.root_P_controlled_feature,
                                                                   self.root_P_reference_feature,
                                                                   cas.Vector3([0, 0, 1])))
        self.expression = cas.logic_and(cas.greater_equal(distance, lower_limit), cas.less_equal(distance, upper_limit))


class AngleMonitor(FeatureMonitor):
    def __init__(self,
                 tip_link: PrefixName,
                 root_link: PrefixName,
                 reference_vector: cas.Vector3,
                 tip_vector: cas.Vector3,
                 lower_angle: float,
                 upper_angle: float,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 pause_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol):
        super().__init__(tip_link=tip_link,
                         root_link=root_link,
                         reference_feature=reference_vector,
                         controlled_feature=tip_vector,
                         name=name,
                         start_condition=start_condition,
                         pause_condition=pause_condition,
                         end_condition=end_condition)

        expr = cas.angle_between_vector(self.root_V_reference_feature, self.root_V_controlled_feature)
        self.expression = cas.logic_and(cas.greater(expr, lower_angle), cas.less(expr, upper_angle))
