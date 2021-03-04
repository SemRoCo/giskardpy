import hashlib

import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from plugin import GiskardBehavior
import giskardpy.identifier as identifier
from giskardpy.tfwrapper import lookup_pose, get_full_frame_name


class WorldVisualizationBehavior(GiskardBehavior):
    def __init__(self, name, ensure_publish=False):
        super(WorldVisualizationBehavior, self).__init__(name)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.marker_namespace = u'planning_world_visualization'
        self.ensure_publish = ensure_publish

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.ids = set()
        return super(WorldVisualizationBehavior, self).setup(timeout)

    def update(self):
        markers = []
        objects = self.get_world().get_objects().values()
        for object in objects:
            for link_name in object.get_link_names():
                if object.has_link_visuals(link_name):
                    marker = object.link_as_marker(link_name)
                    if marker is None:
                        continue
                    marker.header.frame_id = self.map_frame
                    marker.id = int(hashlib.md5(object.get_name() + link_name).hexdigest()[:6],
                                    16)  # FIXME find a better way to give the same link the same id
                    self.ids.add(marker.id)
                    marker.ns = self.marker_namespace
                    marker.header.stamp = rospy.Time()
                    if link_name == object.get_name():
                        marker.pose = object.base_pose
                    else:
                        full_link_name = get_full_frame_name(link_name)
                        if not full_link_name:
                            continue
                        marker.pose = lookup_pose(self.map_frame, full_link_name).pose
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
            marker.ns = self.marker_namespace
            msg.markers.append(marker)
        self.publisher.publish(msg)
        self.ids = set()
