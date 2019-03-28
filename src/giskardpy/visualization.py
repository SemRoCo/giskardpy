import py_trees
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *

from giskardpy.identifier import fk_identifier
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
        for i, link_name in enumerate(self.get_robot().get_link_names()):
            if not self.get_robot().has_link_visuals(link_name):
                continue

            marker = self.get_robot().link_as_marker(link_name)
            if marker is None:
                continue

            marker.scale.x *= 0.99
            marker.scale.y *= 0.99
            marker.scale.z *= 0.99

            marker.header.frame_id = self.robot_base
            marker.action = Marker.ADD
            marker.id = i
            marker.ns = u'planning_visualization'
            marker.header.stamp = rospy.Time()
            marker.pose = self.get_god_map().safe_get_data(fk_identifier + [(self.robot_base, link_name)]).pose
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            markers.append(marker)

        self.publisher.publish(markers)
        return py_trees.common.Status.SUCCESS
