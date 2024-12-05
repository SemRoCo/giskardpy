from copy import deepcopy
from enum import Enum
from typing import Optional, List, Dict, Union

from line_profiler import profile
from tf.transformations import rotation_matrix, quaternion_from_matrix

import giskardpy.casadi_wrapper as cas
import numpy as np
import rospy
from geometry_msgs.msg import Vector3, Point, PoseStamped, Pose, Quaternion
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.data_types.data_types import PrefixName
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions, Collision
import giskardpy_ros.ros1.msg_converter as msg_converter
from giskardpy.model.links import Link
from giskardpy.utils.decorators import memoize, clear_memo
from giskardpy_ros.ros1.ros1_interface import wait_for_publisher, wait_for_topic_to_appear
from giskardpy_ros.ros1.visualization_mode import VisualizationMode
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard
from giskardpy.model.trajectory import Trajectory


class ROSMsgVisualization:
    red = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    yellow = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    green = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    colors = [
        red,  # red
        ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # blue
        yellow,  # yellow
        ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),  # violet
        ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),  # cyan
        green,  # green
        ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),  # white
        ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),  # black
    ]
    mode: VisualizationMode
    frame_locked: bool
    world_version: int

    @profile
    def __init__(self,
                 visualization_topic: str = '~visualization_marker_array',
                 mode: VisualizationMode = VisualizationMode.CollisionsDecomposed):
        self.mode = mode
        self.frame_locked = self.mode in [VisualizationMode.VisualsFrameLocked,
                                          VisualizationMode.CollisionsFrameLocked,
                                          VisualizationMode.CollisionsDecomposedFrameLocked]
        self.publisher = rospy.Publisher(visualization_topic, MarkerArray, queue_size=1, latch=True)
        wait_for_publisher(self.publisher)
        self.marker_ids = {}
        self.tf_root = str(god_map.world.root_link_name)
        GiskardBlackboard().ros_visualizer = self
        self.world_version = -1

    @memoize
    def link_to_marker(self, link: Link) -> List[Marker]:
        return msg_converter.link_to_visualization_marker(data=link, mode=self.mode).markers

    def clear_marker_cache(self) -> None:
        clear_memo(self.link_to_marker)

    def has_world_changed(self) -> bool:
        if self.world_version != god_map.world.model_version:
            self.world_version = god_map.world.model_version
            return True
        return False

    @profile
    def create_world_markers(self, name_space: str = 'world', marker_id_offset: int = 0) -> List[Marker]:
        markers = []
        time_stamp = rospy.Time()
        if self.mode in [VisualizationMode.Visuals, VisualizationMode.VisualsFrameLocked]:
            links = god_map.world.link_names
        else:
            links = god_map.world.link_names_with_collisions
        for i, link_name in enumerate(links):
            link = god_map.world.links[link_name]
            collision_markers = self.link_to_marker(link)
            for j, marker in enumerate(collision_markers):
                if self.frame_locked:
                    marker.header.frame_id = link_name.short_name
                else:
                    marker.header.frame_id = self.tf_root
                marker.action = Marker.ADD
                link_id_key = f'{link_name}_{j}'
                if link_id_key not in self.marker_ids:
                    self.marker_ids[link_id_key] = len(self.marker_ids)
                marker.id = self.marker_ids[link_id_key] + marker_id_offset
                marker.ns = name_space
                marker.header.stamp = time_stamp
                if self.frame_locked:
                    marker.frame_locked = True
                else:
                    marker.pose = god_map.collision_scene.get_map_T_geometry(link_name, j)
                markers.append(marker)
        return markers

    @profile
    def create_collision_markers(self, name_space: str = 'collisions') -> List[Marker]:
        try:
            collisions: Collisions = god_map.closest_point
        except AttributeError as e:
            # no collisions
            return []
        collision_avoidance_configs = god_map.collision_scene.collision_avoidance_configs
        m = Marker()
        m.header.frame_id = self.tf_root
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = name_space
        m.scale = Vector3(x=0.003, y=0.0, z=0.0)
        m.pose.orientation.w = 1.0
        if len(collisions.all_collisions) > 0:
            for collision in collisions.all_collisions:
                group_name = collision.link_a.prefix
                config = collision_avoidance_configs[group_name]
                if collision.is_external:
                    thresholds = config.external_collision_avoidance[collision.link_a]
                else:
                    thresholds = config.self_collision_avoidance[collision.link_a]
                red_threshold = thresholds.hard_threshold
                yellow_threshold = thresholds.soft_threshold
                contact_distance = collision.contact_distance
                if collision.map_P_pa is None:
                    map_T_a = god_map.world.compute_fk_np(god_map.world.root_link_name, collision.original_link_a)
                    map_P_pa = np.dot(map_T_a, collision.a_P_pa)
                else:
                    map_P_pa = collision.map_P_pa

                if collision.map_P_pb is None:
                    map_T_b = god_map.world.compute_fk_np(god_map.world.root_link_name, collision.original_link_b)
                    map_P_pb = np.dot(map_T_b, collision.b_P_pb)
                else:
                    map_P_pb = collision.map_P_pb
                m.points.append(Point(*map_P_pa[:3]))
                m.points.append(Point(*map_P_pb[:3]))
                m.colors.append(self.red)
                m.colors.append(self.green)
                if contact_distance < yellow_threshold:
                    # m.colors[-2] = self.yellow
                    m.colors[-1] = self.yellow
                if contact_distance < red_threshold:
                    # m.colors[-2] = self.red
                    m.colors[-1] = self.red
        else:
            return []
        return [m]

    @profile
    def publish_markers(self, world_ns: str = 'world', collision_ns: str = 'collisions', force: bool = False) -> None:
        if not self.mode == VisualizationMode.Nothing:
            marker_array = MarkerArray()
            if force or (not self.frame_locked or self.frame_locked and self.has_world_changed()):
                self.clear_marker(world_ns)
                marker_array.markers.extend(self.create_world_markers(name_space=world_ns))
            marker_array.markers.extend(self.create_collision_markers(name_space=collision_ns))
            if len(marker_array.markers) > 0:
                self.publisher.publish(marker_array)

    def publish_trajectory_markers(self, trajectory: Trajectory, every_x: int = 10,
                                   start_alpha: float = 0.5, stop_alpha: float = 1.0,
                                   namespace: str = 'trajectory') -> None:
        self.clear_marker(namespace)
        marker_array = MarkerArray()

        def compute_alpha(i):
            if i < 0 or i >= len(trajectory):
                raise ValueError(f'Index {i} is out of range {len(trajectory)}')
            return start_alpha + i * (stop_alpha - start_alpha) / (len(trajectory) - 1)

        with god_map.world.reset_joint_state_context():
            for point_id, joint_state in trajectory.items():
                if point_id % every_x == 0 or point_id == len(trajectory) - 1:
                    god_map.world.state = joint_state
                    god_map.world.notify_state_change()
                    if self.mode not in [VisualizationMode.Visuals, VisualizationMode.VisualsFrameLocked]:
                        god_map.collision_scene.sync()
                    markers = self.create_world_markers(name_space=namespace,
                                                        marker_id_offset=len(marker_array.markers))
                    for m in markers:
                        m.color.a = compute_alpha(point_id)
                    marker_array.markers.extend(deepcopy(markers))
        self.publisher.publish(marker_array)

    def publish_debug_trajectory(self,
                                 debug_expressions: Dict[PrefixName, Union[cas.TransMatrix,
                                 cas.Point3,
                                 cas.Vector3,
                                 cas.Quaternion]],
                                 raw_debug_trajectory: List[Dict[PrefixName, np.ndarray]],
                                 joint_space_traj: Trajectory,
                                 every_x: int = 10,
                                 start_alpha: float = 0.5, stop_alpha: float = 1.0,
                                 namespace: str = 'debug_trajectory') -> None:
        self.clear_marker(namespace)
        marker_array = MarkerArray()

        def compute_alpha(i):
            if i < 0 or i >= len(raw_debug_trajectory):
                raise ValueError("Index i is out of range")
            return start_alpha + i * (stop_alpha - start_alpha) / (len(raw_debug_trajectory) - 1)

        with god_map.world.reset_joint_state_context():
            for point_id, point in enumerate(raw_debug_trajectory):
                joint_state = joint_space_traj.get_exact(point_id)
                god_map.world.state = joint_state
                god_map.world.notify_state_change()
                if self.mode not in [VisualizationMode.Visuals, VisualizationMode.VisualsFrameLocked]:
                    god_map.collision_scene.sync()
                if point_id % every_x == 0 or point_id == len(raw_debug_trajectory) - 1:
                    markers = self.debug_state_to_vectors_markers(debug_expressions=debug_expressions,
                                                                  debug_values=point,
                                                                  marker_id_offset=len(marker_array.markers))
                    for m in markers:
                        m.color.a = compute_alpha(point_id)
                    marker_array.markers.extend(deepcopy(markers))
        self.publisher.publish(marker_array)

    def clear_marker(self, ns: str):
        msg = MarkerArray()
        for i in self.marker_ids.values():
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = i
            marker.ns = ns
            msg.markers.append(marker)
        self.publisher.publish(msg)
        self.marker_ids = {}

    def debug_state_to_vectors_markers(self,
                                       debug_expressions: Dict[PrefixName, Union[cas.TransMatrix,
                                       cas.Point3,
                                       cas.Vector3,
                                       cas.Quaternion]],
                                       debug_values: Dict[PrefixName, np.ndarray],
                                       width: float = 0.05,
                                       marker_id_offset: int = 0) -> List[Marker]:
        ms = []
        color_counter = 0
        for (name, expr), (_, value) in zip(debug_expressions.items(), debug_values.items()):
            if not hasattr(expr, 'reference_frame'):
                continue
            if expr.reference_frame is not None:
                map_T_ref = god_map.world.compute_fk_np(god_map.world.root_link_name, expr.reference_frame)
            else:
                map_T_ref = np.eye(4)
            if isinstance(expr, cas.TransMatrix):
                ref_T_d = value
                map_T_d = np.dot(map_T_ref, ref_T_d)
                map_P_d = map_T_d[:4, 3:]
                # x
                d_V_x_offset = np.array([width, 0, 0, 0])
                map_V_x_offset = np.dot(map_T_d, d_V_x_offset)
                mx = Marker()
                mx.action = mx.ADD
                mx.header.frame_id = self.tf_root
                mx.ns = f'debug/{name}'
                mx.id = 0 + marker_id_offset
                mx.type = mx.CYLINDER
                mx.pose.position.x = map_P_d[0][0] + map_V_x_offset[0]
                mx.pose.position.y = map_P_d[1][0] + map_V_x_offset[1]
                mx.pose.position.z = map_P_d[2][0] + map_V_x_offset[2]
                d_R_x = rotation_matrix(np.pi / 2, [0, 1, 0])
                map_R_x = np.dot(map_T_d, d_R_x)
                mx.pose.orientation = Quaternion(*quaternion_from_matrix(map_R_x))
                mx.color = ColorRGBA(1, 0, 0, 1)
                mx.scale.x = width / 4
                mx.scale.y = width / 4
                mx.scale.z = width * 2
                ms.append(mx)
                # y
                d_V_y_offset = np.array([0, width, 0, 0])
                map_V_y_offset = np.dot(map_T_d, d_V_y_offset)
                my = Marker()
                my.action = my.ADD
                my.header.frame_id = self.tf_root
                my.ns = f'debug/{name}'
                my.id = 1 + marker_id_offset
                my.type = my.CYLINDER
                my.pose.position.x = map_P_d[0][0] + map_V_y_offset[0]
                my.pose.position.y = map_P_d[1][0] + map_V_y_offset[1]
                my.pose.position.z = map_P_d[2][0] + map_V_y_offset[2]
                d_R_y = rotation_matrix(-np.pi / 2, [1, 0, 0])
                map_R_y = np.dot(map_T_d, d_R_y)
                my.pose.orientation = Quaternion(*quaternion_from_matrix(map_R_y))
                my.color = ColorRGBA(0, 1, 0, 1)
                my.scale.x = width / 4
                my.scale.y = width / 4
                my.scale.z = width * 2
                ms.append(my)
                # z
                d_V_z_offset = np.array([0, 0, width, 0])
                map_V_z_offset = np.dot(map_T_d, d_V_z_offset)
                mz = Marker()
                mz.action = mz.ADD
                mz.header.frame_id = self.tf_root
                mz.ns = f'debug/{name}'
                mz.id = 2 + marker_id_offset
                mz.type = mz.CYLINDER
                mz.pose.position.x = map_P_d[0][0] + map_V_z_offset[0]
                mz.pose.position.y = map_P_d[1][0] + map_V_z_offset[1]
                mz.pose.position.z = map_P_d[2][0] + map_V_z_offset[2]
                mz.pose.orientation = Quaternion(*quaternion_from_matrix(map_T_d))
                mz.color = ColorRGBA(0, 0, 1, 1)
                mz.scale.x = width / 4
                mz.scale.y = width / 4
                mz.scale.z = width * 2
                ms.append(mz)
            else:
                m = Marker()
                m.action = m.ADD
                m.ns = f'debug/{name}'
                m.id = 0 + marker_id_offset
                m.header.frame_id = self.tf_root
                m.pose.orientation.w = 1
                if isinstance(expr, cas.Vector3):
                    ref_V_d = value
                    if expr.vis_frame is not None:
                        map_T_vis = god_map.world.compute_fk_np(god_map.world.root_link_name, expr.vis_frame)
                    else:
                        map_T_vis = np.eye(4)
                    map_V_d = np.dot(map_T_ref, ref_V_d)
                    map_P_vis = map_T_vis[:4, 3:].T[0]
                    map_P_p1 = map_P_vis
                    map_P_p2 = map_P_vis + map_V_d * 0.5
                    m.points.append(Point(map_P_p1[0], map_P_p1[1], map_P_p1[2]))
                    m.points.append(Point(map_P_p2[0], map_P_p2[1], map_P_p2[2]))
                    m.type = m.ARROW
                    if expr.color is None:
                        m.color = self.colors[color_counter]
                    else:
                        m.color = expr.color
                    m.scale.x = width / 2
                    m.scale.y = width
                    m.scale.z = 0
                    color_counter += 1
                elif isinstance(expr, cas.Point3):
                    ref_P_d = value
                    map_P_d = np.dot(map_T_ref, ref_P_d)
                    m.pose.position.x = map_P_d[0]
                    m.pose.position.y = map_P_d[1]
                    m.pose.position.z = map_P_d[2]
                    m.pose.orientation.w = 1
                    m.type = m.SPHERE
                    if expr.color is None:
                        m.color = self.colors[color_counter]
                    else:
                        m.color = expr.color
                    m.scale.x = width
                    m.scale.y = width
                    m.scale.z = width
                    color_counter += 1
                ms.append(m)
        return ms
