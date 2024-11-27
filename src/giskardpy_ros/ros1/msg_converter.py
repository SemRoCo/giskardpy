import builtins
import json
from typing import Optional, Union, List, Dict, Any

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from line_profiler import profile
from rospy import Message

import giskardpy.casadi_wrapper as cas
import geometry_msgs.msg as geometry_msgs
import visualization_msgs.msg as visualization_msgs
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import trajectory_msgs.msg as trajectory_msgs
import tf2_msgs.msg as tf2_msgs

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import GiskardError
from giskardpy.data_types.data_types import JointStates, PrefixName, _JointState, ColorRGBA
from giskardpy.data_types.exceptions import GiskardException, CorruptShapeException, UnknownLinkException, \
    UnknownJointException
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import CollisionEntry
from giskardpy.model.joints import MovableJoint
from giskardpy.model.links import LinkGeometry, Link, SphereGeometry, CylinderGeometry, BoxGeometry, MeshGeometry
from giskardpy.model.trajectory import Trajectory
from giskardpy.model.world import WorldTree

from rospy_message_converter.message_converter import \
    convert_dictionary_to_ros_message as original_convert_dictionary_to_ros_message, \
    convert_ros_message_to_dictionary as original_convert_ros_message_to_dictionary

from giskardpy.motion_graph.monitors.monitors import EndMotion, CancelMotion, Monitor
from giskardpy.motion_graph.tasks.task import Task
from giskardpy.utils.math import quaternion_from_rotation_matrix
from giskardpy.utils.utils import get_all_classes_in_module
from giskardpy_ros.ros1.visualization_mode import VisualizationMode


# TODO probably needs some consistency check
# 1. weights are same as in message

def is_ros_message(obj: Any) -> bool:
    return isinstance(obj, Message)


# %% to ros
def to_ros_message(data):
    if isinstance(data, cas.TransMatrix):
        return trans_matrix_to_pose_stamped(data)
    if isinstance(data, cas.Point3):
        return point3_to_point_stamped(data)
    raise ValueError(f'{type(data)} is not a valid type')


def to_visualization_marker(data):
    if isinstance(data, LinkGeometry):
        return link_geometry_to_visualization_marker(data)


@profile
def link_to_visualization_marker(data: Link, mode: VisualizationMode) -> visualization_msgs.MarkerArray:
    markers = visualization_msgs.MarkerArray()
    if mode.is_visual():
        geometries = data.visuals
    else:
        geometries = data.collisions
    for collision in geometries:
        if isinstance(collision, BoxGeometry):
            marker = link_geometry_box_to_visualization_marker(collision)
        elif isinstance(collision, CylinderGeometry):
            marker = link_geometry_cylinder_to_visualization_marker(collision)
        elif isinstance(collision, SphereGeometry):
            marker = link_geometry_sphere_to_visualization_marker(collision)
        elif isinstance(collision, MeshGeometry):
            marker = link_geometry_mesh_to_visualization_marker(collision, mode)
            if mode.is_visual():
                marker.mesh_use_embedded_materials = True
                marker.color = std_msgs.ColorRGBA()
        else:
            raise GiskardException(f'Can\'t convert {type(collision)} to visualization marker.')
        markers.markers.append(marker)
    return markers


@profile
def link_geometry_to_visualization_marker(data: LinkGeometry) -> visualization_msgs.Marker:
    marker = visualization_msgs.Marker()
    marker.color = color_rgba_to_ros_msg(data.color)
    marker.pose = to_ros_message(data.link_T_geometry).pose
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


def link_geometry_mesh_to_visualization_marker(data: MeshGeometry, mode: VisualizationMode) \
        -> visualization_msgs.Marker:
    marker = link_geometry_to_visualization_marker(data)
    marker.type = visualization_msgs.Marker.MESH_RESOURCE
    if mode.is_collision_decomposed():
        marker.mesh_resource = 'file://' + data.collision_file_name_absolute
    else:
        marker.mesh_resource = 'file://' + data.file_name_absolute
    marker.scale.x = data.scale[0]
    marker.scale.y = data.scale[1]
    marker.scale.z = data.scale[2]
    marker.mesh_use_embedded_materials = False
    return marker


def color_rgba_to_ros_msg(data) -> std_msgs.ColorRGBA:
    return std_msgs.ColorRGBA(r=data.r, g=data.g, b=data.b, a=data.a)


def trans_matrix_to_pose_stamped(data: cas.TransMatrix) -> geometry_msgs.PoseStamped:
    pose_stamped = geometry_msgs.PoseStamped()
    pose_stamped.header.frame_id = str(data.reference_frame)
    position = data.to_position().to_np()
    orientation = data.to_rotation().to_quaternion().to_np()
    pose_stamped.pose.position = geometry_msgs.Point(x=position[0],
                                                     y=position[1],
                                                     z=position[2])
    pose_stamped.pose.orientation = geometry_msgs.Quaternion(x=orientation[0],
                                                             y=orientation[1],
                                                             z=orientation[2],
                                                             w=orientation[3])
    return pose_stamped


def numpy_to_pose_stamped(data: np.ndarray, reference_frame: str) -> geometry_msgs.PoseStamped:
    pose_stamped = geometry_msgs.PoseStamped()
    pose_stamped.header.frame_id = str(reference_frame)
    pose_stamped.pose.position.x = data[0, 3]
    pose_stamped.pose.position.y = data[1, 3]
    pose_stamped.pose.position.z = data[2, 3]
    q = quaternion_from_rotation_matrix(data)
    pose_stamped.pose.orientation = geometry_msgs.Quaternion(x=q[0],
                                                             y=q[1],
                                                             z=q[2],
                                                             w=q[3])
    return pose_stamped


def point3_to_point_stamped(data: cas.Point3) -> geometry_msgs.PointStamped:
    point_stamped = geometry_msgs.PointStamped()
    point_stamped.header.frame_id = str(data.reference_frame)
    position = data.to_np()
    point_stamped.point = geometry_msgs.Point(x=position[0], y=position[1], z=position[2])
    return point_stamped


def trans_matrix_to_transform_stamped(data: cas.TransMatrix) -> geometry_msgs.TransformStamped:
    transform_stamped = geometry_msgs.TransformStamped()
    transform_stamped.header.frame_id = data.reference_frame
    transform_stamped.child_frame_id = data.child_frame
    position = data.to_position().to_np()
    orientation = data.to_rotation().to_quaternion().to_np()
    transform_stamped.transform.translation = geometry_msgs.Vector3(x=position[0], y=position[1], z=position[2])
    transform_stamped.transform.rotation = geometry_msgs.Quaternion(x=orientation[0], y=orientation[1],
                                                                    z=orientation[2], w=orientation[3])
    return transform_stamped


def trajectory_to_ros_trajectory(data: Trajectory,
                                 sample_period: float,
                                 start_time: Union[rospy.Duration, float],
                                 joints: List[MovableJoint],
                                 fill_velocity_values: bool = True) -> trajectory_msgs.JointTrajectory:
    if isinstance(start_time, (int, float)):
        start_time = rospy.Duration(start_time)
    trajectory_msg = trajectory_msgs.JointTrajectory()
    trajectory_msg.header.stamp = start_time
    trajectory_msg.joint_names = []
    for i, (time, traj_point) in enumerate(data.items()):
        p = trajectory_msgs.JointTrajectoryPoint()
        p.time_from_start = rospy.Duration(time * sample_period)
        for joint in joints:
            free_variables = joint.get_free_variable_names()
            for free_variable in free_variables:
                if free_variable in traj_point:
                    if i == 0:
                        joint_name = free_variable
                        if isinstance(joint_name, PrefixName):
                            joint_name = joint_name.short_name
                        trajectory_msg.joint_names.append(joint_name)
                    p.positions.append(traj_point[free_variable].position)
                    if fill_velocity_values:
                        p.velocities.append(traj_point[free_variable].velocity)
                else:
                    raise NotImplementedError('generated traj does not contain all joints')
        trajectory_msg.points.append(p)
    return trajectory_msg


@profile
def world_to_tf_message(world: WorldTree, include_prefix: bool) -> tf2_msgs.TFMessage:
    tf_msg = tf2_msgs.TFMessage()
    tf = world._fk_computer.compute_tf()
    current_time = rospy.get_rostime()
    tf_msg.transforms = create_tf_message_batch(len(world._fk_computer.tf))
    for i, (parent_link_name, child_link_name) in enumerate(world._fk_computer.tf):
        pose = tf[i]
        if not include_prefix:
            parent_link_name = parent_link_name.short_name
            child_link_name = child_link_name.short_name

        p_T_c = tf_msg.transforms[i]
        p_T_c.header.frame_id = str(parent_link_name)
        p_T_c.header.stamp = current_time
        p_T_c.child_frame_id = str(child_link_name)
        p_T_c.transform.translation.x = pose[0]
        p_T_c.transform.translation.y = pose[1]
        p_T_c.transform.translation.z = pose[2]
        p_T_c.transform.rotation.x = pose[3]
        p_T_c.transform.rotation.y = pose[4]
        p_T_c.transform.rotation.z = pose[5]
        p_T_c.transform.rotation.w = pose[6]
    return tf_msg


def json_str_to_giskard_kwargs(json_str: str, world: WorldTree) -> Dict[str, Any]:
    ros_kwargs = json_str_to_ros_kwargs(json_str)
    return ros_kwargs_to_giskard_kwargs(ros_kwargs, world)


def json_str_to_ros_kwargs(json_str: str) -> Dict[str, Any]:
    d = json.loads(json_str)
    return json_dict_to_ros_kwargs(d)


def json_dict_to_ros_kwargs(d: Any) -> Dict[str, Any]:
    if isinstance(d, list):
        for i, element in enumerate(d):
            d[i] = json_dict_to_ros_kwargs(element)

    if isinstance(d, dict):
        if 'message_type' in d:
            d = convert_dictionary_to_ros_message(d)
        else:
            for key, value in d.copy().items():
                d[key] = json_dict_to_ros_kwargs(value)
    return d


def ros_kwargs_to_giskard_kwargs(d: Any, world: WorldTree) -> Dict[str, Any]:
    if is_ros_message(d):
        return ros_msg_to_giskard_obj(d, world)
    elif isinstance(d, list):
        for i, element in enumerate(d):
            d[i] = ros_msg_to_giskard_obj(element, world)
    elif isinstance(d, dict):
        for key, value in d.copy().items():
            d[key] = ros_kwargs_to_giskard_kwargs(value, world)
    return d


def convert_dictionary_to_ros_message(json):
    # maybe somehow search for message that fits to structure of json?
    return original_convert_dictionary_to_ros_message(json['message_type'], json['message'])


def monitor_to_ros_msg(monitor: Monitor) -> giskard_msgs.Monitor:
    msg = giskard_msgs.Monitor()
    msg.name = str(monitor.name)
    msg.monitor_class = monitor.__class__.__name__
    msg.start_condition = god_map.monitor_manager.format_condition(monitor.start_condition, new_line=' ')
    msg.kwargs = kwargs_to_json({'hold_condition': god_map.monitor_manager.format_condition(monitor.hold_condition,
                                                                                            new_line=' '),
                                 'end_condition': god_map.monitor_manager.format_condition(monitor.end_condition,
                                                                                           new_line=' ')})
    return msg


def task_to_ros_msg(task: Task) -> giskard_msgs.MotionGoal:
    msg = giskard_msgs.MotionGoal()
    msg.name = str(task.name)
    msg.motion_goal_class = task.__class__.__name__
    msg.start_condition = god_map.monitor_manager.format_condition(task.start_condition, new_line=' ')
    msg.hold_condition = god_map.monitor_manager.format_condition(task.hold_condition, new_line=' ')
    msg.end_condition = god_map.monitor_manager.format_condition(task.end_condition, new_line=' ')
    return msg


def exception_to_error_msg(exception: Exception) -> giskard_msgs.GiskardError:
    error = GiskardError()
    error.type = exception.__class__.__name__
    error.msg = str(exception)
    return error


# %% from ros

exception_classes = get_all_classes_in_module(module_name='giskardpy.data_types.exceptions',
                                              parent_class=GiskardException)

# add all base exceptions
exception_classes.update({name: getattr(builtins, name) for name in dir(builtins) if
                          isinstance(getattr(builtins, name), type) and
                          issubclass(getattr(builtins, name), BaseException)})


def error_msg_to_exception(msg: giskard_msgs.GiskardError) -> Optional[Exception]:
    if msg.type == GiskardError.SUCCESS:
        return None
    if msg.type in exception_classes:
        return exception_classes[msg.type](msg.msg)
    return Exception(f'{msg.type}: {msg.msg}')


def link_name_msg_to_prefix_name(msg: giskard_msgs.LinkName, world: WorldTree) -> PrefixName:
    return world.search_for_link_name(msg.name, msg.group_name)


def joint_name_msg_to_prefix_name(msg: giskard_msgs.LinkName, world: WorldTree) -> PrefixName:
    return world.search_for_joint_name(msg.name, msg.group_name)


def replace_prefix_name_with_str(d: dict) -> dict:
    new_d = d.copy()
    for k, v in d.items():
        if isinstance(k, PrefixName):
            del new_d[k]
            new_d[str(k)] = v
        if isinstance(v, PrefixName):
            new_d[k] = str(v)
        if isinstance(v, dict):
            new_d[k] = replace_prefix_name_with_str(v)
    return new_d


def kwargs_to_json(kwargs: Dict[str, Any]) -> str:
    for k, v in kwargs.copy().items():
        if v is None:
            del kwargs[k]
        else:
            kwargs[k] = thing_to_json(v)
    kwargs = replace_prefix_name_with_str(kwargs)
    return json.dumps(kwargs)


def thing_to_json(thing: Any) -> Any:
    if isinstance(thing, list):
        return [thing_to_json(x) for x in thing]
    if isinstance(thing, dict):
        return {k: thing_to_json(v) for k, v in thing.items()}
    if is_ros_message(thing):
        return convert_ros_message_to_dictionary(thing)
    return thing


def convert_ros_message_to_dictionary(message) -> dict:
    if isinstance(message, list):
        for i, element in enumerate(message):
            message[i] = convert_ros_message_to_dictionary(element)
    elif isinstance(message, dict):
        for k, v in message.copy().items():
            message[k] = convert_ros_message_to_dictionary(v)

    elif isinstance(message, tuple):
        list_values = list(message)
        for i, element in enumerate(list_values):
            list_values[i] = convert_ros_message_to_dictionary(element)
        message = tuple(list_values)

    elif is_ros_message(message):

        type_str_parts = str(type(message)).split('.')
        part1 = type_str_parts[0].split('\'')[1]
        part2 = type_str_parts[-1].split('\'')[0]
        message_type = f'{part1}/{part2}'
        d = {'message_type': message_type,
             'message': original_convert_ros_message_to_dictionary(message)}
        return d

    return message


def msg_type_as_str(msg_type):
    module_str = msg_type.__module__
    parts = module_str.split('.')
    parts[-1] = str(msg_type).split('.')[-1][:-2]
    return '/'.join(parts)


def ros_msg_to_giskard_obj(msg, world: WorldTree):
    if isinstance(msg, sensor_msgs.JointState):
        return ros_joint_state_to_giskard_joint_state(msg)
    elif isinstance(msg, geometry_msgs.PoseStamped):
        return pose_stamped_to_trans_matrix(msg, world)
    elif isinstance(msg, geometry_msgs.Pose):
        return pose_to_trans_matrix(msg)
    elif isinstance(msg, geometry_msgs.PointStamped):
        return point_stamped_to_point3(msg, world)
    elif isinstance(msg, geometry_msgs.Vector3Stamped):
        return vector_stamped_to_vector3(msg, world)
    elif isinstance(msg, geometry_msgs.QuaternionStamped):
        return quaternion_stamped_to_quaternion(msg, world)
    elif isinstance(msg, giskard_msgs.CollisionEntry):
        return collision_entry_msg_to_giskard(msg)
    elif isinstance(msg, giskard_msgs.LinkName):
        try:
            return link_name_msg_to_prefix_name(msg, world)
        except UnknownLinkException as e:
            try:
                return joint_name_msg_to_prefix_name(msg, world)
            except UnknownJointException:
                raise e
    elif isinstance(msg, GiskardError):
        return error_msg_to_exception(msg)
    return msg


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


def world_body_to_link(link_name: PrefixName, msg: giskard_msgs.WorldBody, color: ColorRGBA) -> Link:
    link = Link(link_name)
    geometry = world_body_to_geometry(msg=msg, color=color)
    link.collisions.append(geometry)
    link.visuals.append(geometry)
    return link


def world_body_to_geometry(msg: giskard_msgs.WorldBody, color: ColorRGBA) -> LinkGeometry:
    if msg.type == msg.URDF_BODY:
        raise NotImplementedError()
    elif msg.type == msg.PRIMITIVE_BODY:
        if msg.shape.type == msg.shape.BOX:
            geometry = BoxGeometry(link_T_geometry=cas.TransMatrix(),
                                   depth=msg.shape.dimensions[msg.shape.BOX_X],
                                   width=msg.shape.dimensions[msg.shape.BOX_Y],
                                   height=msg.shape.dimensions[msg.shape.BOX_Z],
                                   color=color)
        elif msg.shape.type == msg.shape.CYLINDER:
            geometry = CylinderGeometry(link_T_geometry=cas.TransMatrix(),
                                        height=msg.shape.dimensions[msg.shape.CYLINDER_HEIGHT],
                                        radius=msg.shape.dimensions[msg.shape.CYLINDER_RADIUS],
                                        color=color)
        elif msg.shape.type == msg.shape.SPHERE:
            geometry = SphereGeometry(link_T_geometry=cas.TransMatrix(),
                                      radius=msg.shape.dimensions[msg.shape.SPHERE_RADIUS],
                                      color=color)
        else:
            raise CorruptShapeException(f'Primitive shape of type {msg.shape.type} not supported.')
    elif msg.type == msg.MESH_BODY:
        if msg.scale.x == 0 or msg.scale.y == 0 or msg.scale.z == 0:
            raise CorruptShapeException(f'Scale of mesh contains 0: {msg.scale}')
        geometry = MeshGeometry(link_T_geometry=cas.TransMatrix(),
                                file_name=msg.mesh,
                                scale=[msg.scale.x, msg.scale.y, msg.scale.z],
                                color=color)
    else:
        raise CorruptShapeException(f'World body type {msg.type} not supported')
    return geometry


def pose_stamped_to_trans_matrix(msg: geometry_msgs.PoseStamped, world: WorldTree) -> cas.TransMatrix:
    p = cas.Point3.from_xyz(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    R = cas.Quaternion.from_xyzw(msg.pose.orientation.x, msg.pose.orientation.y,
                                 msg.pose.orientation.z, msg.pose.orientation.w).to_rotation_matrix()
    result = cas.TransMatrix.from_point_rotation_matrix(point=p,
                                                        rotation_matrix=R,
                                                        reference_frame=world.search_for_link_name(msg.header.frame_id))
    return result


def pose_to_trans_matrix(msg: geometry_msgs.Pose) -> cas.TransMatrix:
    p = cas.Point3.from_xyz(msg.position.x, msg.position.y, msg.position.z)
    R = cas.Quaternion.from_xyzw(msg.orientation.x, msg.orientation.y,
                                 msg.orientation.z, msg.orientation.w).to_rotation_matrix()
    result = cas.TransMatrix.from_point_rotation_matrix(point=p,
                                                        rotation_matrix=R,
                                                        reference_frame=None)
    return result


def point_stamped_to_point3(msg: geometry_msgs.PointStamped, world: WorldTree) -> cas.Point3:
    return cas.Point3.from_xyz(msg.point.x, msg.point.y, msg.point.z,
                               reference_frame=world.search_for_link_name(msg.header.frame_id))


def vector_stamped_to_vector3(msg: geometry_msgs.Vector3Stamped, world: WorldTree) -> cas.Vector3:
    return cas.Vector3.from_xyz(msg.vector.x, msg.vector.y, msg.vector.z,
                                reference_frame=world.search_for_link_name(msg.header.frame_id))


def quaternion_stamped_to_quaternion(msg: geometry_msgs.QuaternionStamped, world: WorldTree) -> cas.RotationMatrix:
    return cas.Quaternion((msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w),
                          reference_frame=world.search_for_link_name(msg.header.frame_id)).to_rotation_matrix()


def collision_entry_msg_to_giskard(msg: giskard_msgs.CollisionEntry) -> CollisionEntry:
    return CollisionEntry(msg.type, msg.distance, msg.group1, msg.group2)


__tf_messages: List[TransformStamped] = [TransformStamped() for _ in range(10000)]


def create_tf_message_batch(size: int) -> List[TransformStamped]:
    global __tf_messages
    return __tf_messages[:size]
