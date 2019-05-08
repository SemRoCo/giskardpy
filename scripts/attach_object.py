#!/usr/bin/env python
import rospy
import sys
from giskardpy.python_interface import GiskardWrapper

if __name__ == '__main__':
    rospy.init_node('attach_object')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        result = giskard.attach_object(name=name, link_frame_id=rospy.get_param('~link'))
        if result.error_codes == result.SUCCESS:
            rospy.loginfo('existing object \'{}\' attached'.format(name))
        else:
            rospy.logwarn('failed to add object \'{}\''.format(name))
            rospy.logwarn(result)
    except KeyError:
        rospy.loginfo('Example call: rosrun giskardpy attach_object.py _name:=box _link:=gripper_tool_frame')
