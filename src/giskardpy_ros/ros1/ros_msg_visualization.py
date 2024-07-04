from typing import Optional, List

import numpy as np
import rospy
from geometry_msgs.msg import Vector3, Point, PoseStamped, Pose
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions, Collision
import giskardpy_ros.ros1.msg_converter as msg_converter
from giskardpy_ros.ros1.ros1_interface import wait_for_publisher, wait_for_topic_to_appear
from giskardpy_ros.tree.blackboard_utils import GiskardBlackboard


class ROSMsgVisualization:
    red = ColorRGBA(1, 0, 0, 1)
    yellow = ColorRGBA(1, 1, 0, 1)
    green = ColorRGBA(0, 1, 0, 1)

    @profile
    def __init__(self, tf_frame: Optional[str] = None, use_decomposed_meshes: bool = True):
        self.use_decomposed_meshes = use_decomposed_meshes
        self.publisher = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1, latch=True)
        wait_for_publisher(self.publisher)
        self.marker_ids = {}
        if tf_frame is None:
            self.tf_root = str(god_map.world.root_link_name)
        else:
            self.tf_root = tf_frame
        GiskardBlackboard().ros_visualizer = self

    @profile
    def create_world_markers(self, name_space: str = 'planning_visualization') -> List[Marker]:
        # todo add caching
        markers = []
        time_stamp = rospy.Time()
        links = god_map.world.link_names_with_collisions
        for i, link_name in enumerate(links):
            link = god_map.world.links[link_name]
            link_markers = msg_converter.link_to_visualization_marker(link, self.use_decomposed_meshes).markers
            for j, marker in enumerate(link_markers):
                marker.header.frame_id = self.tf_root
                marker.action = Marker.ADD
                link_id_key = f'{link_name}_{j}'
                if link_id_key not in self.marker_ids:
                    self.marker_ids[link_id_key] = len(self.marker_ids)
                marker.id = self.marker_ids[link_id_key]
                marker.ns = name_space
                marker.header.stamp = time_stamp
                pose = god_map.collision_scene.get_map_T_geometry(link_name, j)
                if not isinstance(pose, Pose):
                    # TODO handle this better
                    marker.pose = msg_converter.to_ros_message(pose).pose
                else:
                    marker.pose = pose
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
        m.scale = Vector3(0.003, 0, 0)
        m.pose.orientation.w = 1
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
    def publish_markers(self):
        marker_array = MarkerArray()
        marker_array.markers.extend(self.create_world_markers())
        marker_array.markers.extend(self.create_collision_markers())
        self.publisher.publish(marker_array)

    def clear_marker(self):
        msg = MarkerArray()
        for i in self.marker_ids.values():
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = i
            marker.ns = 'planning_visualization'
            msg.markers.append(marker)
        self.publisher.publish(msg)
        self.marker_ids = {}
