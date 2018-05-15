#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point, Quaternion
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest
from giskard_msgs.msg import WorldBody


def add_table_request():
    table = WorldBody()
    table.type = WorldBody.MESH_BODY
    table.name = "table"
    table.pose.header.stamp = rospy.Time.now()
    table.pose.header.frame_id = "map"
    table.pose.pose.position = Point(-1.4, -1.05, 0.0)
    table.pose.pose.orientation = Quaternion(0, 0, 1, 0)
    table.mesh = 'package://iai_kitchen/meshes/misc/big_table_1.stl'
    return UpdateWorldRequest(UpdateWorldRequest.ADD, table)


def add_table(proxy):
    """
    Adds a table to the giskard world, expecting success.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    :return:
    """
    rospy.loginfo("Adding table --expecting success...")
    resp = proxy(add_table_request()) # type: UpdateWorldResponse
    if resp.error_codes != UpdateWorldResponse.SUCCESS:
        raise RuntimeError(resp.error_msg)
    rospy.loginfo("...OK.")


def add_table_again(proxy):
    """
    Adds a table (that is already) to the giskard world, expecting failure.
    :param proxy: ServiceProxy to update the giskard world.
    :type proxy: rospy.ServiceProxy
    :return:
    """
    rospy.loginfo("Adding table, again --expecting failure...")
    resp = proxy(add_table_request()) # type: UpdateWorldResponse
    if resp.error_codes != UpdateWorldResponse.DUPLICATE_BODY_ERROR:
        raise RuntimeError(resp.error_msg)
    rospy.loginfo("...OK.")


def test_update_world():
    rospy.wait_for_service('muh/update_world')
    try:
        update_world = rospy.ServiceProxy('muh/update_world', UpdateWorld)
        add_table(update_world)
        add_table_again(update_world)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == "__main__":
    rospy.init_node('test_update_world')
    rospy.loginfo("Test Update World...")
    test_update_world()