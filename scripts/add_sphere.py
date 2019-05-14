#!/usr/bin/env python
import rospy
import sys
from giskardpy.python_interface import GiskardWrapper
from giskardpy import logging

if __name__ == '__main__':
    rospy.init_node('add_sphere')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        result = giskard.add_sphere(name=name,
                                 size=rospy.get_param('~size', 1),
                                 frame_id=rospy.get_param('~frame', 'map'),
                                 position=rospy.get_param('~position', (0, 0, 0)),
                                 orientation=rospy.get_param('~orientation', (0, 0, 0, 1)))
        if result.error_codes == result.SUCCESS:
            logging.loginfo('sphere \'{}\' added'.format(name))
        else:
            logging.logwarn('failed to add sphere \'{}\''.format(name))
    except KeyError:
        logging.loginfo(
            'Example call: rosrun giskardpy add_sphere.py _name:=sphere _size:=1 _frame_id:=map _position:=[0,0,0] _orientation:=[0,0,0,1]')
