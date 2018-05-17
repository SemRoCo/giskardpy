#!/usr/bin/env python
import rospy
import sys
from giskardpy.python_interface import GiskardWrapper

if __name__ == '__main__':
    # TODO implement if urdf is supported
    rospy.init_node('remove_object')
    giskard = GiskardWrapper([])
    result = giskard.remove_object(sys.argv[1])
    if result.error_codes == result.SUCCESS:
        rospy.loginfo('{} removed'.format(sys.argv[1]))
    else:
        rospy.logwarn('failed to remove {} {}'.format(sys.argv[1], result))
