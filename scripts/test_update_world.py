#!/usr/bin/env python
from time import time

import rospy
import tf2_ros
from geometry_msgs.msg import Point, Vector3, Quaternion, Pose, Transform, PoseStamped, TransformStamped
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from giskard_msgs.msg import WorldBody
from shape_msgs.msg import SolidPrimitive


def to_point_msg(vector3_msg):
    """
    :param vector3_msg:
    :type vector3_msg: Vector3
    :return:
    :rtype: Point
    """
    return Point(vector3_msg.x, vector3_msg.y, vector3_msg.z)


def to_pose_msg(t_msg):
    """

    :param t_msg:
    :type t_msg: Transform
    :return:
    :rtype: Pose
    """
    return Pose(to_point_msg(t_msg.translation), t_msg.rotation)


def to_pose_stamped_msg(t_msg):
    '''

    :param t_msg:
    :type t_msg: TransformStamped
    :return:
    :rtype PoseStamped
    '''
    return PoseStamped(t_msg.header, to_pose_msg(t_msg.transform))


def add_wand_request():
    wand = WorldBody()
    wand.type = WorldBody.PRIMITIVE_BODY
    wand.name = 'wand'
    wand.shape.type = SolidPrimitive.CYLINDER
    wand.shape.dimensions.append(0.15) # height of 15cm
    wand.shape.dimensions.append(0.005) # radius of 0.5cm
    pose = PoseStamped()
    pose.header.frame_id = 'l_gripper_tool_frame'
    pose.pose.orientation.w = 1.0
    return UpdateWorldRequest(UpdateWorldRequest.ADD, wand, True, pose)


def add_table_request():
    table = WorldBody()
    table.type = WorldBody.MESH_BODY
    table.name = "table"
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "map"
    pose.pose.position = Point(-1.4, -1.05, 0.0)
    pose.pose.orientation = Quaternion(0,0,1,0)
    table.mesh = 'package://iai_kitchen/meshes/misc/big_table_1.stl'
    return UpdateWorldRequest(UpdateWorldRequest.ADD, table, False, pose)


def clear_world_request():
    return UpdateWorldRequest(UpdateWorldRequest.REMOVE_ALL, WorldBody(), False, PoseStamped())


def add_cone_request():
    cone = WorldBody()
    cone.type = WorldBody.PRIMITIVE_BODY
    cone.name = 'cone'
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'map'
    pose.pose.orientation = Quaternion(0,0,0,1)
    cone.shape.type = SolidPrimitive.CONE
    cone.shape.dimensions.append(0.01)  # height of 1cm
    cone.shape.dimensions.append(0.05)  # radius of 5cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, cone, False, pose)


def add_sphere_request():
    sphere = WorldBody()
    sphere.type = WorldBody.PRIMITIVE_BODY
    sphere.name = 'sphere'
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'map'
    pose.pose.position = Point(-1.25, -1.0, 0.77)
    pose.pose.orientation = Quaternion(0,0,0,1)
    sphere.shape.type = SolidPrimitive.SPHERE
    sphere.shape.dimensions.append(0.05) # radius of 5cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, sphere, False, pose)


def add_cylinder_request():
    cylinder = WorldBody()
    cylinder.type = WorldBody.PRIMITIVE_BODY
    cylinder.name = 'cylinder'
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'map'
    pose.pose.position = Point(-1.1, -1.0, 0.77)
    pose.pose.orientation = Quaternion(0,0,0,1)
    cylinder.shape.type = SolidPrimitive.CYLINDER
    cylinder.shape.dimensions.append(0.1) # height of 10cm
    cylinder.shape.dimensions.append(0.03)  # radius of 3cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, cylinder, False, pose)


def add_box_request():
    box = WorldBody()
    box.type = WorldBody.PRIMITIVE_BODY
    box.name = 'box'
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = 'map'
    pose.pose.position = Point(-1.5, -1.0, 0.745)
    pose.pose.orientation = Quaternion(0,0,0,1)
    box.shape.type = SolidPrimitive.BOX
    box.shape.dimensions.append(0.2) # X of 20cm
    box.shape.dimensions.append(0.3) # Y of 30cm
    box.shape.dimensions.append(0.05)  # Z of 5cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, box, False, pose)


def rem_sphere_request():
    sphere = WorldBody()
    sphere.name = 'sphere'
    return UpdateWorldRequest(UpdateWorldRequest.REMOVE, sphere, False, PoseStamped())


def add_dummy_pr2_request():
    pr2 = WorldBody()
    pr2.name = "dummy_pr2"
    pr2.type = WorldBody.URDF_BODY
    pr2.urdf = rospy.get_param('/robot_description')
    pr2.joint_state_topic = '/dummy_pr2/joint_states'
    tf = tf2_ros.BufferClient('tf2_buffer_server')
    tf.wait_for_server(rospy.Duration(0.5))
    transform = tf.lookup_transform('map', 'dummy_pr2/base_footprint', rospy.Time.now(), rospy.Duration(0.5))
    return UpdateWorldRequest(UpdateWorldRequest.ADD, pr2, False, to_pose_stamped_msg(transform))

def add_kitchen_request():
    pr2 = WorldBody()
    pr2.name = "kitchen"
    pr2.type = WorldBody.URDF_BODY
    pr2.urdf = rospy.get_param('/kitchen_description')
    pr2.joint_state_topic = '/kitchen_joint_states'
    tf = tf2_ros.BufferClient('tf2_buffer_server')
    tf.wait_for_server(rospy.Duration(0.5))
    transform = tf.lookup_transform('map', 'world', rospy.Time.now(), rospy.Duration(0.5))
    return UpdateWorldRequest(UpdateWorldRequest.ADD, pr2, False, to_pose_stamped_msg(transform))


def call_service(proxy, request, expected_error, user_msg):
    """
        Calls a Service proxy with a given request, and compares result with an expected error. Prints a message to the user.
        :param proxy: ServiceProxy to update the giskard world.
        :type proxy: rospy.ServiceProxy
        :param request: Service request to send to the giskard world.
        :type request: UpdateWorldRequest
        :param expected_error: Error code that we expect to receive.
        :type expected_error: int
        :param user_msg: Message that shall be printed to the user before calling the service.
        :type user_msg: str
    """
    t = time()
    rospy.loginfo(user_msg)
    resp = proxy(request)  # type: UpdateWorldResponse
    if resp.error_codes != expected_error:
        raise RuntimeError(resp.error_msg)
    rospy.loginfo("...OK. Took {}s.".format(time() -t))


def add_wand(proxy):
    """
    Adds a wand into the left gripper of the PR2, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_wand_request(), UpdateWorldResponse.SUCCESS, "Adding wand in gripper --expecting success...")


def add_table_again(proxy):
    """
    Adds a table (that is already) to the giskard world, expecting failure.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_table_request(), UpdateWorldResponse.DUPLICATE_BODY_ERROR, "Adding table, again --expecting failure...")


def add_table(proxy):
    """
    Adds a table to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_table_request(), UpdateWorldResponse.SUCCESS, "Adding table --expecting success...")


def add_sphere(proxy):
    """
    Adds a sphere to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_sphere_request(), UpdateWorldResponse.SUCCESS, "Adding sphere --expecting success...")


def add_cylinder(proxy):
    """
    Adds a cylinder to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_cylinder_request(), UpdateWorldResponse.SUCCESS, "Adding cylinder --expecting success...")

def add_box(proxy):
    """
    Adds a box to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_box_request(), UpdateWorldResponse.SUCCESS, "Adding box --expecting success...")


def add_cone(proxy):
    """
    Adds a cone to the giskard world, expecting failure.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_cone_request(), UpdateWorldResponse.CORRUPT_SHAPE_ERROR, "Adding cone --expecting failure...")


def remove_sphere(proxy):
    """
    Removes sphere from the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, rem_sphere_request(), UpdateWorldResponse.SUCCESS, "Removing sphere --expecting success...")


def remove_sphere_again(proxy):
    """
    Removes non-existing sphere from the giskard world, expecting failure.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, rem_sphere_request(), UpdateWorldResponse.MISSING_BODY_ERROR, "Removing sphere --expecting failure...")


def add_dummy_pr2(proxy):
    """
    Adds dummy PR2 as an object to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_dummy_pr2_request(), UpdateWorldResponse.SUCCESS, "Adding dummy pr2 --expecting success...")

def add_kitchen(proxy):
    """
    Adds dummy PR2 as an object to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_kitchen_request(), UpdateWorldResponse.SUCCESS, "Adding kitchen --expecting success...")


def clear_world(proxy):
    call_service(proxy, clear_world_request(), UpdateWorldResponse.SUCCESS, "Clearing world --expecting success...")


def test_update_world():
    rospy.wait_for_service('giskard/update_world')
    try:
        update_world = rospy.ServiceProxy('giskard/update_world', UpdateWorld)
        # clear_world(update_world)
        # add_table(update_world)
        # add_table_again(update_world)
        # add_sphere(update_world)
        # add_cylinder(update_world)
        # add_box(update_world)
        # add_cone(update_world)
        # remove_sphere(update_world)
        # remove_sphere_again(update_world)
        # add_wand(update_world)
        # add_dummy_pr2(update_world)
        add_kitchen(update_world)
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))


if __name__ == "__main__":
    rospy.init_node('test_update_world')
    rospy.loginfo("Test Update World...")
    test_update_world()