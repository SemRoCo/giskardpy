from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_graph.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.pointing import Pointing
from giskardpy.goals.align_planes import AlignPlanes

from typing import Optional, List, Dict, Union
from std_msgs.msg import ColorRGBA


class FeatureFunctionGoal(Goal):
    def __init__(self,
                 tip_link: str, root_link: str,
                 controlled_feature: Union[PointStamped, Vector3Stamped],
                 reference_feature: Union[PointStamped, Vector3Stamped],
                 name: Optional[str] = None, root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 ):
        self.root = god_map.world.search_for_link_name(root_link, root_group)
        self.tip = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            self.name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        else:
            self.name = name
        super().__init__(self.name)
        reference_feature.header.frame_id = god_map.world.search_for_link_name(reference_feature.header.frame_id,
                                                                               None)
        root_reference_feature = god_map.world.transform_msg(self.root, reference_feature)
        controlled_feature.header.frame_id = god_map.world.search_for_link_name(controlled_feature.header.frame_id,
                                                                                None)
        tip_controlled_feature = god_map.world.transform_msg(self.tip, controlled_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        if isinstance(controlled_feature, PointStamped):
            self.root_P_controlled_feature = root_T_tip.dot(cas.Point3(tip_controlled_feature))
            god_map.debug_expression_manager.add_debug_expression('root_P_controlled_feature',
                                                                  self.root_P_controlled_feature,
                                                                  color=ColorRGBA(r=1, g=0, b=0, a=1))
        elif isinstance(controlled_feature, Vector3Stamped):
            self.root_V_controlled_feature = root_T_tip.dot(cas.Vector3(tip_controlled_feature))
            god_map.debug_expression_manager.add_debug_expression('root_V_controlled_feature',
                                                                  self.root_V_controlled_feature,
                                                                  color=ColorRGBA(r=1, g=0, b=0, a=1))

        if isinstance(reference_feature, PointStamped):
            self.root_P_reference_feature = cas.Point3(root_reference_feature)
            god_map.debug_expression_manager.add_debug_expression('root_P_reference_feature',
                                                                  self.root_P_reference_feature,
                                                                  color=ColorRGBA(r=0, g=1, b=0, a=1))
        if isinstance(reference_feature, Vector3Stamped):
            self.root_V_reference_feature = cas.Vector3(root_reference_feature)
            god_map.debug_expression_manager.add_debug_expression('root_V_reference_feature',
                                                                  self.root_V_reference_feature,
                                                                  color=ColorRGBA(r=0, g=1, b=0, a=1))


class AlignPerpendicular(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_normal: Vector3Stamped,
                 reference_normal: Vector3Stamped,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_normal,
                         controlled_feature=tip_normal, name=name, root_group=root_group, tip_group=tip_group)

        expr = cas.dot(self.root_V_reference_feature[:3], self.root_V_controlled_feature[:3])

        task = self.create_and_add_task()
        task.add_equality_constraint(reference_velocity=max_vel,
                                     equality_bound=0 - expr,
                                     weight=weight,
                                     task_expression=expr,
                                     name=f'{self.name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class HeightGoal(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_point: PointStamped,
                 reference_point: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_point,
                         controlled_feature=tip_point, name=name, root_group=root_group, tip_group=tip_group)

        expr = cas.distance_projected_on_vector(self.root_P_controlled_feature, self.root_P_reference_feature,
                                                cas.Vector3([0, 0, 1]))

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_limit - expr,
                                       lower_error=lower_limit - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{self.name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class DistanceGoal(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_point: PointStamped,
                 reference_point: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_point,
                         controlled_feature=tip_point, name=name, root_group=root_group, tip_group=tip_group)

        projected_vector = cas.distance_vector_projected_on_plane(self.root_P_controlled_feature,
                                                                  self.root_P_reference_feature,
                                                                  cas.Vector3([0, 0, 1]))
        expr = cas.norm(projected_vector)

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_limit - expr,
                                       lower_error=lower_limit - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{self.name}_constraint')
        # An extra constraint that makes the execution more stable
        task.add_inequality_constraint_vector(reference_velocities=[max_vel] * 3,
                                              lower_errors=[0, 0, 0],
                                              upper_errors=[0, 0, 0],
                                              weights=[weight] * 3,
                                              task_expression=projected_vector[:3],
                                              names=[f'{self.name}_extra1', f'{self.name}_extra2', f'{self.name}_extra3'])
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class AngleGoal(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 tip_vector: Vector3Stamped,
                 reference_vector: Vector3Stamped,
                 lower_angle: float,
                 upper_angle: float,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, reference_feature=reference_vector,
                         controlled_feature=tip_vector, name=name, root_group=root_group, tip_group=tip_group)

        expr = cas.angle_between_vector(self.root_V_reference_feature, self.root_V_controlled_feature)

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_angle - expr,
                                       lower_error=lower_angle - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{self.name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
