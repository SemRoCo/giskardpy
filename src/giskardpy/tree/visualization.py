import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.tree.plugin import GiskardBehavior


class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name, ensure_publish=False):
        super(VisualizationBehavior, self).__init__(name)
        self.ensure_publish = ensure_publish
        self.marker_ids = {}

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        return super(VisualizationBehavior, self).setup(timeout)

    def update(self):
        markers = []
        time_stamp = rospy.Time()
        links = self.world.link_names_with_collisions
        for i, link_name in enumerate(links):
            if link_name not in self.world.groups['robot'].link_names:
                for marker in self.world.links[link_name].collision_visualization_markers().markers:
                    marker.header.frame_id = str(self.world.root_link_name)
                    marker.action = Marker.ADD
                    if link_name not in self.marker_ids:
                        self.marker_ids[link_name] = len(self.marker_ids)
                    marker.id = self.marker_ids[link_name]
                    marker.ns = u'planning_visualization'
                    marker.header.stamp = time_stamp
                    marker.pose = self.collision_scene.get_pose(link_name).pose
                    markers.append(marker)

        self.publisher.publish(markers)
        if self.ensure_publish:
            rospy.sleep(0.1)
        return py_trees.common.Status.RUNNING

    def clear_marker(self):
        msg = MarkerArray()
        for i in self.marker_ids.values():
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = i
            marker.ns = u'planning_visualization'
            msg.markers.append(marker)
        self.publisher.publish(msg)
        self.marker_ids = {}
