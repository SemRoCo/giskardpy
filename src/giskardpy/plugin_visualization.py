import hashlib

import py_trees
import rospy
from geometry_msgs.msg import Point, Quaternion
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.tfwrapper import pose_to_kdl, kdl_to_pose
from plugin import GiskardBehavior

class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name, ensure_publish=False):
        super(VisualizationBehavior, self).__init__(name)
        self.ensure_publish = ensure_publish

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.robot_base = self.get_robot().get_root()
        self.ids = set()
        return super(VisualizationBehavior, self).setup(timeout)

    def update(self):
        markers = []
        robot = self.get_robot()
        get_fk = robot.get_fk_pose
        links = [x for x in self.get_robot().get_link_names() if robot.has_link_visuals(x)]
        for i, link_name in enumerate(links):
            marker = robot.link_as_marker(link_name)
            if marker is None:
                continue

            marker.header.frame_id = self.robot_base
            marker.action = Marker.ADD
            marker.id = int(hashlib.md5(link_name).hexdigest()[:6],
                            16)  # FIXME find a better way to give the same link the same id
            self.ids.add(marker.id)
            marker.ns = u'planning_visualization'
            marker.header.stamp = rospy.Time()

            origin = robot.get_urdf_link(link_name).visual.origin
            fk = get_fk(self.robot_base, link_name).pose

            if origin is not None:
                marker.pose.position = Point(*origin.xyz)
                marker.pose.orientation = Quaternion(*quaternion_from_euler(*origin.rpy))
                marker.pose = kdl_to_pose(pose_to_kdl(fk) * pose_to_kdl(marker.pose))
            else:
                marker.pose = fk
            markers.append(marker)

        self.publisher.publish(markers)
        if self.ensure_publish:
            rospy.sleep(0.1)
        return py_trees.common.Status.SUCCESS

    def clear_marker(self):
        msg = MarkerArray()
        for i in self.ids:
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = i
            marker.ns = u'planning_visualization'
            msg.markers.append(marker)
        self.publisher.publish(msg)
        self.ids = set()
