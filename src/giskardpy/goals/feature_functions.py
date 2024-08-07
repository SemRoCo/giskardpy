from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.pointing import Pointing
from giskardpy.goals.align_planes import AlignPlanes

from typing import Optional, List, Dict, Union
from std_msgs.msg import ColorRGBA


class FeatureFunctionGoal(Goal):
    def __init__(self,
                 tip_link: str, root_link: str,
                 world_feature: Union[PointStamped, Vector3Stamped],
                 robot_feature: Union[PointStamped, Vector3Stamped],
                 name: Optional[str] = None
                 ):
        self.root = god_map.world.search_for_link_name(root_link, None)
        self.tip = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root}/{self.tip}'
        super().__init__(name)
        world_feature.header.frame_id = god_map.world.search_for_link_name(world_feature.header.frame_id, None)
        root_world_feature = god_map.world.transform_msg(self.root, world_feature)
        robot_feature.header.frame_id = god_map.world.search_for_link_name(robot_feature.header.frame_id, None)
        tip_robot_feature = god_map.world.transform_msg(self.tip, robot_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root, self.tip)
        if type(robot_feature) == PointStamped:
            self.root_P_robot_feature = root_T_tip.dot(cas.Point3(tip_robot_feature))
            god_map.debug_expression_manager.add_debug_expression('root_P_robot_feature',
                                                                  self.root_P_robot_feature,
                                                                  color=ColorRGBA(r=1, g=0, b=0, a=1))
        elif type(robot_feature) == Vector3Stamped:
            self.root_V_robot_feature = root_T_tip.dot(cas.Vector3(tip_robot_feature))
            god_map.debug_expression_manager.add_debug_expression('root_V_robot_feature',
                                                                  self.root_V_robot_feature,
                                                                  color=ColorRGBA(r=1, g=0, b=0, a=1))

        if type(world_feature) == PointStamped:
            self.root_P_world_feature = cas.Point3(root_world_feature)
            god_map.debug_expression_manager.add_debug_expression('root_P_world_feature',
                                                                  self.root_P_world_feature,
                                                                  color=ColorRGBA(r=0, g=1, b=0, a=1))
        if type(world_feature) == Vector3Stamped:
            self.root_V_world_feature = cas.Vector3(root_world_feature)
            god_map.debug_expression_manager.add_debug_expression('root_V_world_feature',
                                                                  self.root_V_world_feature,
                                                                  color=ColorRGBA(r=0, g=1, b=0, a=1))


class PerpendicularFeatureFunction(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: Vector3Stamped,
                 robot_feature: Vector3Stamped,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name)

        expr = cas.dot(self.root_V_world_feature[:3], self.root_V_robot_feature[:3])

        task = self.create_and_add_task()
        task.add_equality_constraint(reference_velocity=max_vel,
                                     equality_bound=0 - expr,
                                     weight=weight,
                                     task_expression=expr,
                                     name=f'{name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class HeightFeatureFunction(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: PointStamped,
                 robot_feature: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name)

        distance = self.root_P_robot_feature - self.root_P_world_feature

        height_vector = cas.Vector3([0, 0, 1])

        projection = cas.dot(distance, height_vector)
        expr = projection

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_limit - expr,
                                       lower_error=lower_limit - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class DistanceFeatureFunction(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: PointStamped,
                 robot_feature: PointStamped,
                 lower_limit: float,
                 upper_limit: float,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name)

        # distance between the two feature points
        distance = self.root_P_robot_feature - self.root_P_world_feature

        # normal vector defining the x-y plane
        height_vector = cas.Vector3([0, 0, 1])

        # project the distance vector onto the x-y plane and calculate its length
        projection = cas.norm(distance - cas.dot(distance, height_vector) * height_vector)
        expr = projection
        projected_vector = distance - cas.dot(distance, height_vector) * height_vector

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_limit - expr,
                                       lower_error=lower_limit - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{name}_constraint')
        task.add_inequality_constraint_vector(reference_velocities=[max_vel] * 3,
                                              lower_errors=[0, 0, 0],
                                              upper_errors=[0, 0, 0],
                                              weights=[weight] * 3,
                                              task_expression=projected_vector[:3],
                                              names=['dsf', 'sdf', 'fdg'])
        # god_map.debug_expression_manager.add_debug_expression('distance-expr', expr)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class PointingFeatureFunction(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: PointStamped,
                 robot_feature: Vector3Stamped,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        self.add_constraints_of_goal(Pointing(tip_link=tip_link,
                                              goal_point=world_feature,
                                              root_link=root_link,
                                              pointing_axis=robot_feature,
                                              max_velocity=max_vel,
                                              weight=weight,
                                              name=name,
                                              start_condition=start_condition,
                                              hold_condition=hold_condition,
                                              end_condition=end_condition))


class AlignFeatureFunction(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: Vector3Stamped,
                 robot_feature: Vector3Stamped,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)

        self.add_constraints_of_goal(AlignPlanes(root_link=root_link,
                                                 tip_link=tip_link,
                                                 goal_normal=world_feature,
                                                 tip_normal=robot_feature,
                                                 reference_velocity=max_vel,
                                                 weight=weight,
                                                 name=name,
                                                 start_condition=start_condition,
                                                 hold_condition=hold_condition,
                                                 end_condition=end_condition))


class AngleFeatureFunction(FeatureFunctionGoal):
    def __init__(self, tip_link: str, root_link: str,
                 world_feature: Vector3Stamped,
                 robot_feature: Vector3Stamped,
                 lower_angle: float,
                 upper_angle: float,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol
                 ):
        super().__init__(tip_link=tip_link, root_link=root_link, world_feature=world_feature,
                         robot_feature=robot_feature, name=name)

        expr = cas.angle_between_vector(self.root_V_world_feature, self.root_V_robot_feature)

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_angle - expr,
                                       lower_error=lower_angle - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
