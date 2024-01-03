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
import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy.utils.tfwrapper import normalize_quaternion_msg, np_to_kdl, point_to_kdl, kdl_to_point, \
    quaternion_to_kdl, transform_to_kdl, kdl_to_transform_stamped


class DebugMarkerPublisher(GiskardBehavior):
    colors = [ColorRGBA(0, 0, 0, 1),  # black
              ColorRGBA(1, 0, 0, 1),  # red
              ColorRGBA(0, 1, 0, 1),  # green
              ColorRGBA(1, 1, 0, 1),  # yellow
              ColorRGBA(0, 0, 1, 1),  # blue
              ColorRGBA(1, 0, 1, 1),  # violet
              ColorRGBA(0, 1, 1, 1),  # cyan
              ColorRGBA(1, 1, 1, 1)]  # white

    @profile
    def __init__(self, name, tf_topic='/tf', map_frame: Optional[str] = None):
        super().__init__(name)
        if map_frame is None:
            self.map_frame = str(self.world.root_link_name)
        else:
            self.map_frame = map_frame
        self.tf_pub = rospy.Publisher(tf_topic, TFMessage, queue_size=10)
        self.marker_pub = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=10)

    @record_time
    def setup(self, timeout):
        self.clear_markers()

    def publish_debug_markers(self):
        ms = MarkerArray()
        ms.markers.extend(self.to_vectors_markers())
        self.marker_pub.publish(ms)

    def to_vectors_markers(self, width: float = 0.05) -> List[Marker]:
        ms = []
        color_counter = 0
        for name, value in self.debugs_evaluated.items():
            expr = self.debugs[name]
            if not hasattr(expr, 'reference_frame'):
                continue
            if expr.reference_frame is not None:
                map_T_ref = self.world.compute_fk_np(self.world.root_link_name, expr.reference_frame)
            else:
                map_T_ref = np.eye(4)
            if isinstance(expr, w.TransMatrix):
                ref_T_d = value
                map_T_d = np.dot(map_T_ref, ref_T_d)
                map_P_d = map_T_d[:4, 3:]
                # x
                d_V_x_offset = np.array([width, 0, 0, 0])
                map_V_x_offset = np.dot(map_T_d, d_V_x_offset)
                mx = Marker()
                mx.action = mx.ADD
                mx.header.frame_id = self.map_frame
                mx.ns = f'debug{name}'
                mx.id = 0
                mx.type = mx.CYLINDER
                mx.pose.position.x = map_P_d[0][0] + map_V_x_offset[0]
                mx.pose.position.y = map_P_d[1][0] + map_V_x_offset[1]
                mx.pose.position.z = map_P_d[2][0] + map_V_x_offset[2]
                d_R_x = rotation_matrix(np.pi / 2, [0, 1, 0])
                map_R_x = np.dot(map_T_d, d_R_x)
                mx.pose.orientation = Quaternion(*quaternion_from_matrix(map_R_x))
                mx.color = ColorRGBA(1, 0, 0, 1)
                mx.scale.x = width / 4
                mx.scale.y = width / 4
                mx.scale.z = width * 2
                ms.append(mx)
                # y
                d_V_y_offset = np.array([0, width, 0, 0])
                map_V_y_offset = np.dot(map_T_d, d_V_y_offset)
                my = Marker()
                my.action = my.ADD
                my.header.frame_id = self.map_frame
                my.ns = f'debug{name}'
                my.id = 1
                my.type = my.CYLINDER
                my.pose.position.x = map_P_d[0][0] + map_V_y_offset[0]
                my.pose.position.y = map_P_d[1][0] + map_V_y_offset[1]
                my.pose.position.z = map_P_d[2][0] + map_V_y_offset[2]
                d_R_y = rotation_matrix(-np.pi / 2, [1, 0, 0])
                map_R_y = np.dot(map_T_d, d_R_y)
                my.pose.orientation = Quaternion(*quaternion_from_matrix(map_R_y))
                my.color = ColorRGBA(0, 1, 0, 1)
                my.scale.x = width / 4
                my.scale.y = width / 4
                my.scale.z = width * 2
                ms.append(my)
                # z
                d_V_z_offset = np.array([0, 0, width, 0])
                map_V_z_offset = np.dot(map_T_d, d_V_z_offset)
                mz = Marker()
                mz.action = mz.ADD
                mz.header.frame_id = self.map_frame
                mz.ns = f'debug{name}'
                mz.id = 2
                mz.type = mz.CYLINDER
                mz.pose.position.x = map_P_d[0][0] + map_V_z_offset[0]
                mz.pose.position.y = map_P_d[1][0] + map_V_z_offset[1]
                mz.pose.position.z = map_P_d[2][0] + map_V_z_offset[2]
                mz.pose.orientation = Quaternion(*quaternion_from_matrix(map_T_d))
                mz.color = ColorRGBA(0, 0, 1, 1)
                mz.scale.x = width / 4
                mz.scale.y = width / 4
                mz.scale.z = width * 2
                ms.append(mz)
            else:
                m = Marker()
                m.action = m.ADD
                m.ns = f'debug/{name}'
                m.id = 0
                m.header.frame_id = self.map_frame
                m.pose.orientation.w = 1
                if isinstance(expr, w.Vector3):
                    ref_V_d = value
                    if expr.vis_frame is not None:
                        map_T_vis = self.world.compute_fk_np(self.world.root_link_name, expr.vis_frame)
                    else:
                        map_T_vis = np.eye(4)
                    map_V_d = np.dot(map_T_ref, ref_V_d)
                    map_P_vis = map_T_vis[:4, 3:].T[0]
                    map_P_p1 = map_P_vis
                    map_P_p2 = map_P_vis + map_V_d
                    m.points.append(Point(map_P_p1[0], map_P_p1[1], map_P_p1[2]))
                    m.points.append(Point(map_P_p2[0], map_P_p2[1], map_P_p2[2]))
                    m.type = m.ARROW
                    m.color = self.colors[color_counter]
                    m.scale.x = width / 2
                    m.scale.y = width
                    m.scale.z = 0
                    color_counter += 1
                elif isinstance(expr, w.Point3):
                    ref_P_d = value
                    map_P_d = np.dot(map_T_ref, ref_P_d)
                    m.pose.position.x = map_P_d[0]
                    m.pose.position.y = map_P_d[1]
                    m.pose.position.z = map_P_d[2]
                    m.pose.orientation.w = 1
                    m.type = m.SPHERE
                    m.color = self.colors[color_counter]
                    m.scale.x = width
                    m.scale.y = width
                    m.scale.z = width
                    color_counter += 1
                ms.append(m)
        return ms

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_pub.publish(msg)

    @record_time
    @profile
    def update(self):
        with self.god_map as god_map:
            self.debugs = self.god_map.unsafe_get_data(identifier.debug_expressions)
            if len(self.debugs) > 0:
                self.debugs_evaluated = self.god_map.unsafe_get_data(identifier.debug_expressions_evaluated)
                self.publish_debug_markers()
        return Status.RUNNING
