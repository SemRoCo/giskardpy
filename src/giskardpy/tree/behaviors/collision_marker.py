from typing import List, Set, Union

import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from py_trees import Status
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.identifier as identifier
from giskardpy.model.collision_world_syncer import Collision, Collisions
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class CollisionMarker(GiskardBehavior):
    red = ColorRGBA(1, 0, 0, 1)
    yellow = ColorRGBA(1, 1, 0, 1)
    green = ColorRGBA(0, 1, 0, 1)

    @profile
    def __init__(self, name):
        super().__init__(name)
        self.map_frame = str(self.world.root_link_name)

    @record_time
    @profile
    def setup(self, timeout=10.0, name_space='pybullet_collisions'):
        super().setup(timeout)
        self.pub_collision_marker = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=10)
        self.name_space = name_space
        return True

    @record_time
    @profile
    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        collisions = self.get_god_map().get_data(identifier.closest_point)
        if len(collisions.all_collisions) > 0:
            self.publish_cpi_markers(collisions)
        return Status.SUCCESS

    def collision_to_marker(self, collisions: Union[Set[Collision], List[Collision]]) -> Marker:
        m = Marker()
        m.header.frame_id = self.map_frame
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = self.name_space
        m.scale = Vector3(0.003, 0, 0)
        m.pose.orientation.w = 1
        if len(collisions) > 0:
            for collision in collisions:  # type: Collision
                group_name = collision.link_a.prefix
                config = self.collision_avoidance_configs[group_name]
                if collision.is_external:
                    thresholds = config.external_collision_avoidance[collision.link_a]
                else:
                    thresholds = config.self_collision_avoidance[collision.link_a]
                red_threshold = thresholds.hard_threshold
                yellow_threshold = thresholds.soft_threshold
                contact_distance = collision.contact_distance
                if collision.map_P_pa is None:
                    map_T_a = self.world.compute_fk_np(self.world.root_link_name, collision.original_link_a)
                    map_P_pa = np.dot(map_T_a, collision.a_P_pa)
                else:
                    map_P_pa = collision.map_P_pa

                if collision.map_P_pb is None:
                    map_T_b = self.world.compute_fk_np(self.world.root_link_name, collision.original_link_b)
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
        return m

    def publish_cpi_markers(self, collisions: Collisions):
        """
        Publishes a string for each ClosestPointInfo in the dict. If the distance is below the threshold, the string
        is colored red. If it is below threshold*2 it is yellow. If it is below threshold*3 it is green.
        Otherwise no string will be published.
        :type collisions: Collisions
        """
        m = self.collision_to_marker(collisions.items())
        ma = MarkerArray()
        ma.markers.append(m)
        if len(ma.markers[0].points) > 0:
            self.pub_collision_marker.publish(ma)
            pass
