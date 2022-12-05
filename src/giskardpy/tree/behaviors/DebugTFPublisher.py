from typing import Dict, List

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Quaternion, Point, Transform
from py_trees import Status
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_matrix
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

import giskardpy.casadi_wrapper as w
import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import normalize_quaternion_msg, np_to_kdl, point_to_kdl, kdl_to_point, \
    quaternion_to_kdl, transform_to_kdl, kdl_to_transform_stamped


class DebugTFPublisher(GiskardBehavior):
    colors = [ColorRGBA(0, 0, 0, 1),
              ColorRGBA(1, 0, 0, 1),
              ColorRGBA(0, 1, 0, 1),
              ColorRGBA(1, 1, 0, 1),
              ColorRGBA(0, 0, 1, 1),
              ColorRGBA(1, 0, 1, 1),
              ColorRGBA(0, 1, 1, 1),
              ColorRGBA(1, 1, 1, 1)]

    @profile
    def __init__(self, name, tf_topic='/tf'):
        super().__init__(name)
        self.tf_pub = rospy.Publisher(tf_topic, TFMessage, queue_size=10)
        self.marker_pub = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=10)

    def setup(self, timeout):
        self.clear_markers()

    def publish_debug_markers(self):
        ms = MarkerArray()
        ms.markers.extend(self.to_vectors_markers())
        self.marker_pub.publish(ms)

    def to_vectors_markers(self, width: float = 0.02) -> List[Marker]:
        ms = []
        for i, (name, ref_V_d) in enumerate(self.debugs_evaluated.items()):
            expr = self.debugs[name]
            if isinstance(expr, w.Vector3):
                m = Marker()
                m.header.frame_id = 'map'
                if expr.reference_frame is not None:
                    map_T_ref = self.world.compute_fk_np(self.world.root_link_name, expr.reference_frame)
                    map_T_vis = self.world.compute_fk_np(self.world.root_link_name, expr.vis_frame)
                    map_V_d = np.dot(map_T_ref, ref_V_d)
                    map_P_vis = map_T_vis[:4, 3:]
                    map_P_p1 = map_P_vis
                    map_P_p2 = map_P_vis + map_V_d
                    m.points.append(Point(map_P_p1[0][0], map_P_p1[1][0], map_P_p1[2][0]))
                    m.points.append(Point(map_P_p2[0][0], map_P_p2[1][0], map_P_p2[2][0]))
                else:
                    map_V_d = ref_V_d
                    m.points.append(Point())
                    m.points.append(Point(map_V_d[0][0], map_V_d[1][0], map_V_d[2][0]))
                m.action = m.ADD
                m.ns = f'debug/{name}'
                m.id = 0
                m.type = m.ARROW
                m.color = self.colors[i]
                m.scale.x = width
                m.scale.y = width
                m.scale.z = 0
                ms.append(m)
        return ms

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_pub.publish(msg)

    @profile
    def update(self):
        with self.get_god_map() as god_map:
            self.debugs_evaluated = self.god_map.unsafe_get_data(identifier.debug_expressions_evaluated)
            self.debugs = self.god_map.unsafe_get_data(identifier.debug_expressions)
            if len(self.debugs) > 0:
                self.publish_debug_markers()
        return Status.RUNNING
