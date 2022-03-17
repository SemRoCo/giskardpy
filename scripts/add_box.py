#!/usr/bin/env python
import rospy
from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils import logging

if __name__ == '__main__':
    rospy.init_node('add_box')
    giskard = GiskardWrapper()
    try:
        name = rospy.get_param('~name')
        result = giskard.add_box(name=name,
                                 size=rospy.get_param('~size', (1, 1, 1)),
                                 frame_id=rospy.get_param('~frame_id', 'map'),
                                 position=rospy.get_param('~position', (0, 0, 0)),
                                 orientation=rospy.get_param('~orientation', (0, 0, 0, 1)))
        rospy.sleep(0.5)
        if result.error_codes == result.SUCCESS:
            logging.loginfo('box \'{}\' added'.format(name))
        else:
            logging.logwarn('failed to add box \'{}\''.format(name))
    except KeyError:
        logging.loginfo(
            'Example call: rosrun giskardpy add_box.py _name:=box _size:=[1,1,1] _frame_id:=map _position:=[0,0,0] _orientation:=[0,0,0,1]')
