#!/usr/bin/env python
import rospy
import sys
from giskardpy.python_interface import GiskardWrapper

if __name__ == '__main__':
    rospy.init_node('add_cylinder')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        result = giskard.add_cylinder(name=name,
                                 size=rospy.get_param('~size', (1, 1)),
                                 frame_id=rospy.get_param('~frame', 'map'),
                                 position=rospy.get_param('~position', (0, 0, 0)),
                                 orientation=rospy.get_param('~orientation', (0, 0, 0, 1)))
        if result.error_codes == result.SUCCESS:
            rospy.loginfo('cylinder \'{}\' added'.format(name))
        else:
            rospy.logwarn('failed to add cylinder \'{}\''.format(name))
    except KeyError:
        rospy.loginfo(
            'Example call: rosrun giskardpy add_cylinder.py _name:=cylinder _size:=[1,1] _frame_id:=map _position:=[0,0,0] _orientation:=[0,0,0,1]')
