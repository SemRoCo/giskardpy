import hashlib

import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.plugin import GiskardBehavior
import giskardpy.identifier as identifier
from giskardpy.tfwrapper import lookup_pose, get_full_frame_name, pose_to_kdl, kdl_to_pose


class WorldVisualizationBehavior(GiskardBehavior):
    def __init__(self, name, ensure_publish=False):
        super(WorldVisualizationBehavior, self).__init__(name)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.marker_namespace = u'planning_world_visualization'
        self.ensure_publish = ensure_publish
        self.currently_publishing_objects = {}
        self.links_full_frame_name = {}

    def setup(self, timeout):
        self.publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.ids = set()
        self.set_full_link_names_for_objects()
        return super(WorldVisualizationBehavior, self).setup(timeout)

    def set_full_link_names_for_objects(self):
        """
        Resets the full link names (namespace name + link name) for every current
        object in the world and saves the object names of these objects.
        """
        self.currently_publishing_objects = {}
        self.links_full_frame_name = {}
        objects_dict = self.get_world().get_objects()
        for object_name, object in objects_dict.items():
            self.currently_publishing_objects[object_name] = object
            for link_name in object.get_link_names():
                if object.has_link_visuals(link_name):
                    try:
                        self.links_full_frame_name[link_name] = get_full_frame_name(link_name)
                    except KeyError:
                        continue

    def get_id_str(self, object_name, link_name):
        return '{}{}'.format(object_name, link_name).encode('utf-8')

    def has_environment_changed(self):
        """
        Checks if objects in the world were added or removed and if so it returns True. Otherwise, False.
        """
        objects_dict = self.get_world().get_objects()
        object_names = [object_name for object_name, _ in objects_dict.items()]
        curr_publishing_object_names = [object_name for object_name, _ in self.currently_publishing_objects.items()]
        return object_names != curr_publishing_object_names

    def update(self):
        markers = []
        time_stamp = rospy.Time()
        objects_dict = self.get_world().get_objects()

        # If objects were added, update the namespace and objects
        if self.has_environment_changed():
            self.set_full_link_names_for_objects()

        # Creating of marker for every link in an object
        for object_name, object in objects_dict.items():
            for link_name in object.get_link_names():
                if object.has_link_visuals(link_name):
                    # Simple objects (containing only one link) are published here:
                    if link_name == object_name and len(object.get_link_names()) == 1:
                        marker = object.as_marker_msg()
                        markers.append(marker)
                        continue
                    # More complex objects will be published here:
                    else:
                        marker = object.link_as_marker(link_name)
                        if marker is None:
                            continue
                        marker.header.frame_id = self.map_frame
                        id_str = self.get_id_str(object_name, link_name)
                        marker.id = int(hashlib.md5(id_str).hexdigest()[:6],
                                        16)  # FIXME find a better way to give the same link the same id
                        self.ids.add(marker.id)
                        marker.ns = self.marker_namespace
                        marker.header.stamp = time_stamp
                        try:
                            full_link_name = self.links_full_frame_name[link_name]
                        except KeyError:
                            continue
                        pose = lookup_pose(self.map_frame, full_link_name).pose
                        if object.has_non_identity_visual_offset(link_name):
                            marker.pose = kdl_to_pose(pose_to_kdl(pose) * pose_to_kdl(marker.pose))
                        else:
                            marker.pose = pose
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