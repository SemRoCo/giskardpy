from __future__ import division

from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.pointing import Pointing

from typing import Optional, List, Dict


class PerpendicularFeatureFunction(Goal):
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
        world_feature.header.frame_id = god_map.world.search_for_link_name(world_feature.header.frame_id, None)
        root_world_feature = god_map.world.transform_msg(self.root_link, world_feature)
        robot_feature.header.frame_id = god_map.world.search_for_link_name(robot_feature.header.frame_id, None)
        tip_robot_feature = god_map.world.transform_msg(self.tip_link, robot_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_V_robot_feature = root_T_tip.dot(cas.Vector3(tip_robot_feature))
        root_V_world_feature = cas.Vector3(root_world_feature)

        expr = cas.dot(root_V_world_feature[:3], root_V_robot_feature[:3])

        task = self.create_and_add_task()
        task.add_equality_constraint(reference_velocity=max_vel,
                                     equality_bound=0 - expr,
                                     weight=weight,
                                     task_expression=expr,
                                     name=f'{name}_constraint')
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class HeightFeatureFunction(Goal):
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
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        world_feature.header.frame_id = god_map.world.search_for_link_name(world_feature.header.frame_id, None)
        root_world_feature = god_map.world.transform_msg(self.root_link, world_feature)
        robot_feature.header.frame_id = god_map.world.search_for_link_name(robot_feature.header.frame_id, None)
        tip_robot_feature = god_map.world.transform_msg(self.tip_link, robot_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_P_robot_feature = root_T_tip.dot(cas.Point3(tip_robot_feature))
        root_P_world_feature = cas.Point3(root_world_feature)

        distance = root_P_robot_feature - root_P_world_feature

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


class DistanceFeatureFunction(Goal):
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
        self.root_link = god_map.world.search_for_link_name(root_link, None)
        self.tip_link = god_map.world.search_for_link_name(tip_link, None)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        world_feature.header.frame_id = god_map.world.search_for_link_name(world_feature.header.frame_id, None)
        root_world_feature = god_map.world.transform_msg(self.root_link, world_feature)
        robot_feature.header.frame_id = god_map.world.search_for_link_name(robot_feature.header.frame_id, None)
        tip_robot_feature = god_map.world.transform_msg(self.tip_link, robot_feature)

        root_T_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link)
        root_P_robot_feature = root_T_tip.dot(cas.Point3(tip_robot_feature))
        root_P_world_feature = cas.Point3(root_world_feature)

        # distance between the two feature points
        distance = root_P_robot_feature - root_P_world_feature

        # normal vector defining the x-y plane
        height_vector = cas.Vector3([0, 0, 1])

        # project the distance vector onto the x-y plane and calculate its length
        projection = cas.norm(distance - cas.dot(distance, height_vector) * height_vector)
        expr = projection

        task = self.create_and_add_task()
        task.add_inequality_constraint(reference_velocity=max_vel,
                                       upper_error=upper_limit - expr,
                                       lower_error=lower_limit - expr,
                                       weight=weight,
                                       task_expression=expr,
                                       name=f'{name}_constraint')
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
