#!/usr/bin/env python

import rospy
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse

def test_update_world():
    rospy.wait_for_service('muh/update_world')
    try:
        update_world = rospy.ServiceProxy('muh/update_world', UpdateWorld)
        resp = update_world() # type: UpdateWorldResponse
        return resp
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == "__main__":
    print "Test Update World"
    print "{}".format(test_update_world())