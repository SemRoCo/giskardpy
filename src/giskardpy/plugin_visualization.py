import py_trees
import rospy
from geometry_msgs.msg import Point, Quaternion
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.identifier import fk_identifier
from giskardpy.tfwrapper import pose_to_kdl, kdl_to_pose
from plugin import GiskardBehavior


class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name, enable_visualization):
        super(VisualizationBehavior, self).__init__(name)
        self.enable_visualization = enable_visualization

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.robot_base = self.get_robot().get_root()
        return True

    def update(self):
        if not self.enable_visualization:
            return py_trees.common.Status.SUCCESS

        markers = []
        links = [x for x in self.get_robot().get_link_names() if self.get_robot().has_link_visuals(x)]
        for i, link_name in enumerate(links):
            marker = self.get_robot().link_as_marker(link_name)
            if marker is None:
                continue

            marker.scale.x *= 0.99
            marker.scale.y *= 0.99
            marker.scale.z *= 0.99

            marker.header.frame_id = self.robot_base
            marker.action = Marker.ADD
            marker.id = i
            marker.ns = 'planning_visualization'
            marker.header.stamp = rospy.Time()

            origin = self.get_robot().get_urdf_link(link_name).visual.origin
            fk = self.get_god_map().safe_get_data(fk_identifier + [(self.robot_base, link_name)]).pose
            if origin is not None:
                marker.pose.position = Point(*origin.xyz)
                marker.pose.orientation = Quaternion(*quaternion_from_euler(*origin.rpy))
                marker.pose = kdl_to_pose(pose_to_kdl(fk) * pose_to_kdl(marker.pose))
            else:
                marker.pose = fk
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            markers.append(marker)

        self.publisher.publish(markers)
        return py_trees.common.Status.SUCCESS
