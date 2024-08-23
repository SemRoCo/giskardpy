from copy import deepcopy
from enum import Enum
from typing import Optional, List

import numpy as np
import rospy
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions, Collision
from giskardpy.model.trajectory import Trajectory


class VisualizationMode(Enum):
    Nothing = -1
    Visuals = 0
    Collisions = 1
    CollisionsDecomposed = 2
    VisualsFrameLocked = 3
    CollisionsFrameLocked = 4
    CollisionsDecomposedFrameLocked = 5


class ROSMsgVisualization:
    red = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    yellow = ColorRGBA(1.0, 1.0, 0.0, 1.0)
    green = ColorRGBA(0.0, 1.0, 0.0, 1.0)
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
        self.publisher = rospy.Publisher(visualization_topic, MarkerArray, queue_size=1)
        self.marker_ids = {}
        self.tf_root = str(god_map.world.root_link_name)
        god_map.ros_visualizer = self
        self.world_version = -1

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
            if self.mode in [VisualizationMode.Visuals,
                             VisualizationMode.VisualsFrameLocked]:
                collision_markers = link.visuals_visualization_markers().markers
            elif self.mode in [VisualizationMode.Collisions,
                               VisualizationMode.CollisionsFrameLocked]:
                collision_markers = link.collision_visualization_markers(use_decomposed_meshes=False).markers
            elif self.mode in [VisualizationMode.CollisionsDecomposed,
                               VisualizationMode.CollisionsDecomposedFrameLocked]:
                collision_markers = link.collision_visualization_markers(use_decomposed_meshes=True).markers
            else:
                raise AttributeError(f'Invalid visualization mode: {self.mode}')
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
        m.scale = Vector3(0.003, 0.0, 0.0)
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
            self.publisher.publish(marker_array)

    def publish_trajectory_markers(self, trajectory: Trajectory, every_x: int = 10,
                                   start_alpha: float = 0.5, stop_alpha: float = 1.0,
                                   namespace: str = 'trajectory') -> None:
        self.clear_marker(namespace)
        marker_array = MarkerArray()

        def compute_alpha(i):
            if i < 0 or i >= len(trajectory):
                raise ValueError("Index i is out of range")
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
