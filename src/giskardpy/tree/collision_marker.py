from collections import defaultdict
import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from py_trees import Status
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.identifier as identifier
from giskardpy.data_types import Collision, Collisions, KeyDefaultDict
from giskardpy.tree.plugin import GiskardBehavior


class CollisionMarker(GiskardBehavior):
    def setup(self, timeout=10.0, name_space='pybullet_collisions'):
        super(CollisionMarker, self).setup(timeout)
        self.pub_collision_marker = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1)
        self.name_space = name_space
        rospy.sleep(.5)
        return True

    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        collisions = self.get_god_map().get_data(identifier.closest_point)
        for robot_name in self.collision_scene.robot_names:
            if len(collisions[robot_name].all_collisions) > 0:
                self.publish_cpi_markers(collisions[robot_name])
        return Status.RUNNING

    def publish_cpi_markers(self, collisions):
        """
        Publishes a string for each ClosestPointInfo in the dict. If the distance is below the threshold, the string
        is colored red. If it is below threshold*2 it is yellow. If it is below threshold*3 it is green.
        Otherwise no string will be published.
        :type collisions: Collisions
        """
        m = Marker()
        m.header.frame_id = str(self.world.root_link_name)
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        m.ns = self.name_space
        m.scale = Vector3(0.003, 0, 0)
        m.pose.orientation.w = 1
        if len(collisions.items()) > 0:
            for collision in collisions.items():  # type: Collision
                red_threshold = 0.05  # TODO don't hardcode this
                yellow_threshold = red_threshold * 2
                green_threshold = yellow_threshold * 2
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
                if contact_distance < green_threshold:
                    m.points.append(Point(*map_P_pa[:3]))
                    m.points.append(Point(*map_P_pb[:3]))
                    m.colors.append(ColorRGBA(0, 1, 0, 1))
                    m.colors.append(ColorRGBA(0, 1, 0, 1))
                if contact_distance < yellow_threshold:
                    m.colors[-2] = ColorRGBA(1, 1, 0, 1)
                    m.colors[-1] = ColorRGBA(1, 1, 0, 1)
                if contact_distance < red_threshold:
                    m.colors[-2] = ColorRGBA(1, 0, 0, 1)
                    m.colors[-1] = ColorRGBA(1, 0, 0, 1)
        ma = MarkerArray()
        ma.markers.append(m)
        if len(ma.markers[0].points) > 0:
            self.pub_collision_marker.publish(ma)
            pass
