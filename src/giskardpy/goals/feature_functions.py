from __future__ import division

import rospy
import tf
from geometry_msgs.msg import PointStamped, PoseStamped, QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from giskardpy import casadi_wrapper as cas
from giskardpy.goals.goal import Goal
from giskardpy.god_map import god_map
from giskardpy.symbol_manager import symbol_manager
from giskardpy.motion_graph.tasks.task import Task, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.goals.pointing import Pointing
from giskardpy.goals.align_planes import AlignPlanes
import numpy as np

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
            self.root_V_controlled_feature.vis_frame = controlled_feature.header.frame_id
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
            self.root_V_reference_feature.vis_frame = controlled_feature.header.frame_id
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
                                              names=[f'{self.name}_extra1', f'{self.name}_extra2',
                                                     f'{self.name}_extra3'])
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


class MovementFunctionGoal(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        pass


class MixingMovementFunction(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 start_position: PointStamped,
                 plane_normal: Vector3Stamped = None,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 radius_growth: float = 0.01,
                 angle_growth: float = 1,
                 max_radius: float = 0.1,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, root_group)
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        god_map.world.transform_msg(root_link, start_position)
        start_time = god_map.monitor_manager.register_expression_updater(symbol_manager.time + 0, start_condition)

        def define_spiral(a=radius_growth, b=angle_growth, max_radius=max_radius):
            # Get the time symbol from the symbol manager
            t = symbol_manager.time - start_time

            # Define the radius as a function of time (linear growth)
            r = cas.limit(a * t, 0, max_radius)

            # Define the angle as a function of time (spiral angle increases with time)
            theta = b * t

            # Parametric equations for a spiral
            x = r * cas.cos(theta)
            y = r * cas.sin(theta)

            # Return the x and y coordinates as the spiral function
            return x, y

        task = self.create_and_add_task(task_name='Mixing Movement')
        x, y = define_spiral()

        if plane_normal is None:
            plane_normal = Vector3Stamped()
            plane_normal.header.frame_id = 'map'
            plane_normal.vector.z = 1

        root_plane_normal = god_map.world.transform_msg(self.root_link, plane_normal)
        normal = cas.Vector3(root_plane_normal)
        if root_plane_normal.vector.z == 1:
            function = cas.Point3(start_position) + cas.Vector3([x, y, 0])
        else:
            u = cas.cross(cas.Vector3([0, 0, 1]), normal)
            u = u / cas.norm(u)
            v = cas.cross(normal, u)
            rot = cas.TransMatrix(cas.hstack([u, v, normal, cas.Point3([0, 0, 0])]))
            function = cas.Point3(start_position) + cas.dot(rot, cas.Vector3([x, y, 0]))

        root_P_tip = god_map.world.compose_fk_expression(self.root_link, self.tip_link).to_position()
        task.add_equality_constraint_vector(reference_velocities=[max_vel] * 3,
                                            equality_bounds=(function - root_P_tip)[:3],
                                            weights=[weight] * 3,
                                            task_expression=root_P_tip[:3],
                                            names=['x', 'y', 'z'])
        god_map.debug_expression_manager.add_debug_expression('spiral', function)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)


class ForceConstraint(Goal):
    def __init__(self, tip_link: str, root_link: str,
                 threshold: float,
                 sensor_topic: str,
                 max_vel: float = 0.2,
                 root_group: Optional[str] = None,
                 tip_group: Optional[str] = None,
                 name: str = None,
                 weight: int = WEIGHT_BELOW_CA,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.FalseSymbol
                 ):
        self.root_link = god_map.world.search_for_link_name(root_link, root_group)
        self.tip_link = god_map.world.search_for_link_name(tip_link, tip_group)
        if name is None:
            name = f'{self.__class__.__name__}/{self.root_link}/{self.tip_link}'
        super().__init__(name)
        self.listener = tf.TransformListener()
        self.force_vector_buffer = []  # Buffer to store the last N values
        self.window_size = 30
        self.force_vector = None
        rospy.Subscriber(name=sensor_topic, data_class=Vector3Stamped, callback=self.callback)

        root_T_tip = god_map.world.compose_fk_expression(root_link=self.root_link, tip_link=self.tip_link)
        tip_V_force = symbol_manager.get_expr(
            f'god_map.motion_goal_manager.motion_goals[\'{str(self)}\'].force_vector', output_type_hint=cas.Vector3,
            input_type_hint=list)
        root_V_force = cas.dot(root_T_tip, tip_V_force)
        root_V_force[2] = 0
        intensity = cas.norm(root_V_force)
        reduce_direction = root_V_force / intensity

        dist = reduce_direction.dot(root_T_tip.to_position())

        task = self.create_and_add_task('forceConstraint')
        task.add_inequality_constraint_vector(reference_velocities=[max_vel] * 3,
                                              lower_errors=reduce_direction[:3] * 0.01,
                                              upper_errors=reduce_direction[:3] * 0.01,
                                              weights=[cas.if_greater(intensity, threshold, weight, 0)] * 3,
                                              task_expression=root_T_tip.to_position()[:3],
                                              names=['x', 'y', 'z'])
        # task.add_inequality_constraint(reference_velocity=max_vel,
        #                                lower_error=threshold - intensity,
        #                                upper_error=float('inf'),
        #                                weight=weight,
        #                                task_expression=dist)

        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)
        reduce_direction.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression('reduce-direction', reduce_direction)
        god_map.debug_expression_manager.add_debug_expression('intensity', intensity)

    def callback(self, data: Vector3Stamped):
        # data.header.frame_id = 'tool'
        # data = self.listener.transformVector3('hand_palm_link', data)
        force_vector = np.array([data.vector.x, data.vector.y, data.vector.z])

        # Append the new force vector to the buffer
        self.force_vector_buffer.append(force_vector)

        # If the buffer exceeds the window size, remove the oldest value
        if len(self.force_vector_buffer) > self.window_size:
            self.force_vector_buffer.pop(0)

        # Calculate the moving average over the buffer
        self.force_vector = np.mean(self.force_vector_buffer, axis=0)
