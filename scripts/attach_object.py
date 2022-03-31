#!/usr/bin/env python
import rospy
from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils import logging

if __name__ == '__main__':
    rospy.init_node('attach_object')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        result = giskard.update_parent_link_of_group(name=name, link_frame_id=rospy.get_param('~link'))
        if result.error_codes == result.SUCCESS:
            logging.loginfo('existing object \'{}\' attached'.format(name))
        else:
            logging.logwarn('failed to add object \'{}\''.format(name))
            logging.logwarn(result)
    except KeyError:
        logging.loginfo('Example call: rosrun giskardpy attach_object.py _name:=box _link:=gripper_tool_frame')
