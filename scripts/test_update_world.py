#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point, Quaternion
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from giskard_msgs.msg import WorldBody
from shape_msgs.msg import SolidPrimitive


def add_table_request(position=(-1.4, -1.05, 0.0), orientation=(0,0,1,0)):
    table = WorldBody()
    table.type = WorldBody.MESH_BODY
    table.name = "table"
    table.pose.header.stamp = rospy.Time.now()
    table.pose.header.frame_id = "map"
    table.pose.pose.position = Point(*position)
    table.pose.pose.orientation = Quaternion(*orientation)
    table.mesh = 'package://iai_kitchen/meshes/misc/big_table_1.stl'
    return UpdateWorldRequest(UpdateWorldRequest.ADD, table)


def clear_world_request():
    return UpdateWorldRequest(UpdateWorldRequest.REMOVE_ALL, WorldBody())


def add_sphere_request():
    sphere = WorldBody()
    sphere.type = WorldBody.PRIMITIVE_BODY
    sphere.name = 'sphere'
    sphere.pose.header.stamp = rospy.Time.now()
    sphere.pose.header.frame_id = 'map'
    sphere.pose.pose.position = Point(-1.3, -1.0, 0.77)
    sphere.pose.pose.orientation = Quaternion(0,0,0,1)
    sphere.shape.type = SolidPrimitive.SPHERE
    sphere.shape.dimensions.append(0.05) # radius of 5cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, sphere)


def add_cylinder_request():
    cylinder = WorldBody()
    cylinder.type = WorldBody.PRIMITIVE_BODY
    cylinder.name = 'cylinder'
    cylinder.pose.header.stamp = rospy.Time.now()
    cylinder.pose.header.frame_id = 'map'
    cylinder.pose.pose.position = Point(-1.1, -1.0, 0.77)
    cylinder.pose.pose.orientation = Quaternion(0,0,0,1)
    cylinder.shape.type = SolidPrimitive.CYLINDER
    cylinder.shape.dimensions.append(0.1) # height of 10cm
    cylinder.shape.dimensions.append(0.03)  # radius of 3cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, cylinder)


def add_box_request():
    box = WorldBody()
    box.type = WorldBody.PRIMITIVE_BODY
    box.name = 'box'
    box.pose.header.stamp = rospy.Time.now()
    box.pose.header.frame_id = 'map'
    box.pose.pose.position = Point(-1.5, -1.0, 0.745)
    box.pose.pose.orientation = Quaternion(0,0,0,1)
    box.shape.type = SolidPrimitive.BOX
    box.shape.dimensions.append(0.2) # X of 20cm
    box.shape.dimensions.append(0.3) # Y of 30cm
    box.shape.dimensions.append(0.05)  # Z of 5cm
    return UpdateWorldRequest(UpdateWorldRequest.ADD, box)


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
    rospy.loginfo(user_msg)
    resp = proxy(request)  # type: UpdateWorldResponse
    if resp.error_codes != expected_error:
        raise RuntimeError(resp.error_msg)
    rospy.loginfo("...OK.")


def add_table_again(proxy):
    """
    Adds a table (that is already) to the giskard world, expecting failure.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_table_request(), UpdateWorldResponse.DUPLICATE_BODY_ERROR, "Adding table, again --expecting failure...")


def add_table(proxy, position=(-1.4, -1.05, 0.0), orientation=(0,0,1,0)):
    """
    Adds a table to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    """
    call_service(proxy, add_table_request(position, orientation), UpdateWorldResponse.SUCCESS, "Adding table --expecting success...")


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


def clear_world(proxy):
    call_service(proxy, clear_world_request(), UpdateWorldResponse.SUCCESS, "Clearing world --expecting success...")


def test_update_world():
    rospy.wait_for_service('muh/update_world')
    try:
        update_world = rospy.ServiceProxy('muh/update_world', UpdateWorld)
        clear_world(update_world)
        add_table(update_world)
        add_table_again(update_world)
        add_sphere(update_world)
        add_cylinder(update_world)
        add_box(update_world)
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))


if __name__ == "__main__":
    rospy.init_node('test_update_world')
    rospy.loginfo("Test Update World...")
    test_update_world()