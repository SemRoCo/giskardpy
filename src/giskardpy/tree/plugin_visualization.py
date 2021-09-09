import hashlib

import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import pose_to_kdl, kdl_to_pose

class VisualizationBehavior(GiskardBehavior):
    def __init__(self, name, ensure_publish=False):
        super(VisualizationBehavior, self).__init__(name)
        self.ensure_publish = ensure_publish

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.robot_base = self.get_robot().root_link_name
        self.ids = set()
        return super(VisualizationBehavior, self).setup(timeout)

    def update(self):
        markers = []
        time_stamp = rospy.Time()
        robot = self.get_robot()
        get_fk = robot.compute_fk_pose
        links = [x for x in self.get_robot().link_names if robot.links[x].has_visual()]
        for i, link_name in enumerate(links):
            marker = robot.link_as_marker(link_name)
            if marker is None:
                continue

            marker.header.frame_id = self.robot_base
            marker.action = Marker.ADD
            marker.id = int(hashlib.md5(link_name.encode('utf-8')).hexdigest()[:6],
                            16)  # FIXME find a better way to give the same link the same id
            self.ids.add(marker.id)
            marker.ns = u'planning_visualization'
            marker.header.stamp = time_stamp

            fk = get_fk(self.robot_base, link_name).pose

            if robot.has_non_identity_visual_offset(link_name):
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
