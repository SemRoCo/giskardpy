#!/usr/bin/env python
import rospy
import sys
from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils import logging

if __name__ == '__main__':
    rospy.init_node('remove_object')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
    except KeyError:
        try:
            name = sys.argv[1]
        except IndexError:
            logging.loginfo('Example call: rosrun giskardpy remove_object.py _name:=muh')
            sys.exit()
    result = giskard.remove_group(name)
    if result.error_codes == result.SUCCESS:
        logging.loginfo('{} removed'.format(name))
    else:
        logging.logwarn('failed to remove {} {}'.format(name, result))
