import PyKDL
import rospy
from geometry_msgs.msg import Point, Vector3
from py_trees import Status
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.identifier as identifier
from giskardpy.data_types import ClosestPointInfo, Collisions
from giskardpy.plugin import GiskardBehavior
from giskardpy.tfwrapper import msg_to_kdl
import numpy as np

class CPIMarker(GiskardBehavior):
    def setup(self, timeout=10.0):
        super(CPIMarker, self).setup(timeout)
        self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        rospy.sleep(.5)
        return True

    def update(self):
        """
        Computes closest point info for all robot links and safes it to the god map.
        """
        collisions = self.get_god_map().get_data(identifier.closest_point)
        if collisions:
            self.publish_cpi_markers(collisions)
        return Status.SUCCESS

    def publish_cpi_markers(self, collisions):
        """
        Publishes a string for each ClosestPointInfo in the dict. If the distance is below the threshold, the string
        is colored red. If it is below threshold*2 it is yellow. If it is below threshold*3 it is green.
        Otherwise no string will be published.
        :type collisions: Collisions
        """
        m = Marker()
        m.header.frame_id = self.get_robot().get_root()
        m.action = Marker.ADD
        m.type = Marker.LINE_LIST
        m.id = 1337
        # TODO make namespace parameter
        m.ns = u'pybullet collisions'
        m.scale = Vector3(0.003, 0, 0)
        if len(collisions.items()) > 0:
            for collision_info in collisions.items():  # type: list
                # if len(collision_infos) > 0:
                #     collision_info = collision_infos[0]
                    red_threshold = collision_info.min_dist
                    yellow_threshold = red_threshold * 2
                    green_threshold = red_threshold * 3

                    if collision_info.contact_distance < green_threshold:
                        root_T_link = self.get_robot().get_fk_np(self.get_robot().get_root(), collision_info.frame)
                        a__root = np.dot(root_T_link, np.concatenate((collision_info.position_on_a, [1])))[:-1]
                        m.points.append(Point(*a__root))
                        m.points.append(Point(*collision_info.position_on_b))
                        m.colors.append(ColorRGBA(0, 1, 0, 1))
                        m.colors.append(ColorRGBA(0, 1, 0, 1))
                    if collision_info.contact_distance < yellow_threshold:
                        m.colors[-2] = ColorRGBA(1, 1, 0, 1)
                        m.colors[-1] = ColorRGBA(1, 1, 0, 1)
                    if collision_info.contact_distance < red_threshold:
                        m.colors[-2] = ColorRGBA(1, 0, 0, 1)
                        m.colors[-1] = ColorRGBA(1, 0, 0, 1)
        ma = MarkerArray()
        ma.markers.append(m)
        self.pub_collision_marker.publish(ma)
