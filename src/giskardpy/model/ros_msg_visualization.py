from typing import Optional

import rospy
from visualization_msgs.msg import MarkerArray, Marker

from giskardpy import identifier
from giskardpy.god_map import GodMap


class ROSMsgVisualization:
    @profile
    def __init__(self, tf_frame: Optional[str] = None):
        self.publisher = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1)
        self.marker_ids = {}
        self.god_map = GodMap()
        self.world = self.god_map.get_data(identifier.world)
        if tf_frame is None:
            self.tf_root = str(self.world.root_link_name)
        else:
            self.tf_root = tf_frame
        self.god_map.set_data(identifier.ros_visualizer, self)
        self.collision_scene = self.god_map.get_data(identifier.collision_scene)

    @profile
    def publish_markers(self):
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
