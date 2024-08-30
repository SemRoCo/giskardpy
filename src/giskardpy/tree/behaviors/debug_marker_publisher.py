from typing import Dict, List, Optional

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, Point, Transform
from py_trees import Status
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_matrix, quaternion_about_axis, rotation_matrix
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.casadi_wrapper as w
from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.utils.tfwrapper import normalize_quaternion_msg, np_to_kdl, point_to_kdl, kdl_to_point, \
    quaternion_to_kdl, transform_to_kdl, kdl_to_transform_stamped


class DebugMarkerPublisher(GiskardBehavior):


    @profile
    def __init__(self, name: str = 'debug marker', tf_topic: str = '/tf', map_frame: Optional[str] = None):
        super().__init__(name)
        if map_frame is None:
            self.map_frame = str(god_map.world.root_link_name)
        else:
            self.map_frame = map_frame
        self.tf_pub = rospy.Publisher(tf_topic, TFMessage, queue_size=10)
        self.marker_pub = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=10)

    @record_time
    def setup(self, timeout):
        self.clear_markers()
        return super().setup(timeout)

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_pub.publish(msg)

    @record_time
    @profile
    def update(self):
        debug_exprs = god_map.debug_expression_manager.debug_expressions
        if len(debug_exprs) > 0:
            debug_state = god_map.debug_expression_manager.evaluated_debug_expressions
            ms = MarkerArray()
            markers = god_map.ros_visualizer.debug_state_to_vectors_markers(debug_exprs, debug_state)
            ms.markers.extend(markers)
            self.marker_pub.publish(ms)
        return Status.SUCCESS
