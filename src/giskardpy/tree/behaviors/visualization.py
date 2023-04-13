import py_trees
import rospy
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class VisualizationBehavior(GiskardBehavior):
    @profile
    def __init__(self, name, ensure_publish=False):
        super().__init__(name)
        self.ensure_publish = ensure_publish
        self.marker_ids = {}
        self.tf_root = str(self.world.root_link_name)

    @profile
    def setup(self, timeout):
        self.publisher = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1)
        return super().setup(timeout)

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        markers = []
        time_stamp = rospy.Time()
        links = self.world.link_names_with_collisions
        for i, link_name in enumerate(links):
            for j, marker in enumerate(self.world.links[link_name].collision_visualization_markers().markers):
                marker.header.frame_id = self.tf_root
                marker.action = Marker.ADD
                link_id_key = f'{link_name}_{j}'
                if link_id_key not in self.marker_ids:
                    self.marker_ids[link_id_key] = len(self.marker_ids)
                marker.id = self.marker_ids[link_id_key]
                marker.ns = 'planning_visualization'
                marker.header.stamp = time_stamp
                marker.pose = self.collision_scene.get_pose(link_name, j).pose
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
            marker.ns = 'planning_visualization'
            msg.markers.append(marker)
        self.publisher.publish(msg)
        self.marker_ids = {}
