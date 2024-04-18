from typing import Optional

import giskardpy.casadi_wrapper as cas
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs

from giskardpy.data_types.data_types import JointStates, PrefixName, _JointState
from giskardpy.exceptions import GiskardException
from giskardpy.model.links import LinkGeometry, Link, SphereGeometry, CylinderGeometry, BoxGeometry, MeshGeometry


# %% to ros
def to_ros_message(data):
    if isinstance(data, cas.TransMatrix):
        return trans_matrix_to_pose_stamped(data)


def to_visualization_marker(data):
    if isinstance(data, LinkGeometry):
        return link_geometry_to_visualization_marker(data)


def link_to_visualization_marker(data: Link, use_decomposed_meshes: bool) -> visualization_msgs.MarkerArray:
    markers = visualization_msgs.MarkerArray()
    for collision in data.collisions:
        if isinstance(collision, BoxGeometry):
            marker = link_geometry_box_to_visualization_marker(collision)
        elif isinstance(collision, CylinderGeometry):
            marker = link_geometry_cylinder_to_visualization_marker(collision)
        elif isinstance(collision, SphereGeometry):
            marker = link_geometry_sphere_to_visualization_marker(collision)
        elif isinstance(collision, MeshGeometry):
            marker = link_geometry_mesh_to_visualization_marker(collision, use_decomposed_meshes)
        else:
            raise GiskardException('failed conversion')
        markers.markers.append(marker)
    return markers


def link_geometry_to_visualization_marker(data: LinkGeometry) -> visualization_msgs.Marker:
    marker = visualization_msgs.Marker()
    marker.color = color_rgba_to_ros_msg(data.color)
    return marker


def link_geometry_sphere_to_visualization_marker(data: SphereGeometry) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.SPHERE
    marker.scale.x = data.radius * 2
    marker.scale.y = data.radius * 2
    marker.scale.z = data.radius * 2
    return marker


def link_geometry_cylinder_to_visualization_marker(data: CylinderGeometry) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.CYLINDER
    marker.scale.x = data.radius * 2
    marker.scale.y = data.radius * 2
    marker.scale.z = data.height
    return marker


def link_geometry_box_to_visualization_marker(data: BoxGeometry) -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.CUBE
    marker.scale.x = data.depth
    marker.scale.y = data.width
    marker.scale.z = data.height
    return marker


def link_geometry_mesh_to_visualization_marker(data: MeshGeometry, use_decomposed_meshes: bool) \
        -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.MESH_RESOURCE
    if use_decomposed_meshes:
        marker.mesh_resource = 'file://' + data.collision_file_name_absolute
    else:
        marker.mesh_resource = 'file://' + data.file_name_absolute
    marker.scale.x = data.scale[0]
    marker.scale.y = data.scale[1]
    marker.scale.z = data.scale[2]
    marker.mesh_use_embedded_materials = False
    return marker


def color_rgba_to_ros_msg(data) -> std_msgs.ColorRGBA:
    return std_msgs.ColorRGBA(data.r, data.g, data.b, data.a)


def trans_matrix_to_pose_stamped(data: cas.TransMatrix) -> geometry_msgs.PoseStamped:
    pose_stamped = geometry_msgs.PoseStamped()
    pose_stamped.header.frame_id = data.reference_frame
    position = data.to_position().to_np()
    orientation = data.to_rotation().to_quaternion().to_np()
    pose_stamped.pose.position = geometry_msgs.Point(position[0][0], position[1][0], position[2][0])
    pose_stamped.pose.orientation = geometry_msgs.Quaternion(orientation[0][0], orientation[1][0],
                                                             orientation[2][0], orientation[3][0])
    return pose_stamped


# %% from ros
def convert_ros_msg(msg):
    if isinstance(msg, sensor_msgs.JointState):
        return ros_joint_state_to_giskard_joint_state(msg)


def ros_joint_state_to_giskard_joint_state(msg: sensor_msgs.JointState, prefix: Optional[str] = None) -> JointStates:
    js = JointStates()
    for i, joint_name in enumerate(msg.name):
        joint_name = PrefixName(joint_name, prefix)
        sjs = _JointState(position=msg.position[i],
                          velocity=0,
                          acceleration=0,
                          jerk=0,
                          snap=0,
                          crackle=0,
                          pop=0)
        js[joint_name] = sjs
    return js
