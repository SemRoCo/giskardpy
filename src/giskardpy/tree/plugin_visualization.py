import hashlib

import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import pose_to_kdl, kdl_to_pose

class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name, ensure_publish=False, clear=False):
        super(VisualizationBehavior, self).__init__(name)
        self.ensure_publish = ensure_publish
        self.clear = clear
        self.marker_ids = {}

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.robot_base = self.get_robot().root_link_name
        self.ids = set()
        return super(VisualizationBehavior, self).setup(timeout)

    def update(self):
        if self.clear:
            self.clear_marker()
        markers = []
        time_stamp = rospy.Time()
        get_fk = self.world.compute_fk_pose
        links = self.world.link_names_with_collisions
        for i, link_name in enumerate(links):
            for marker in self.world.links[link_name].collision_visualization_markers().markers:
                marker.header.frame_id = str(self.world.root_link_name)
                marker.action = Marker.ADD
                if link_name not in self.marker_ids:
                    self.marker_ids[link_name] = len(self.marker_ids)
                marker.id = self.marker_ids[link_name]
                self.ids.add(marker.id)
                marker.ns = u'planning_visualization'
                marker.header.stamp = time_stamp

                fk = get_fk(self.robot_base, link_name).pose

                marker.pose = kdl_to_pose(pose_to_kdl(fk) * pose_to_kdl(marker.pose))
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
        self.ids = set()
