#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy
import numpy as np

from giskardpy.python_interface.old_python_interface import OldGiskardWrapper

from giskardpy.utils import logging


class MUH:
    cancel_msg = 'Canceling all Giskard goals.'

    def __init__(self, button_ids):
        self.giskard = OldGiskardWrapper()
        joy_msg: Joy = rospy.wait_for_message('/joy', Joy)
        self.button_filter = np.zeros(len(joy_msg.buttons), dtype=bool)
        self.button_filter[button_ids] = True
        self.button_ids = button_ids
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_cb)

    def joy_cb(self, joy_msg: Joy):
        buttons = np.array(joy_msg.buttons)
        filtered_buttons = buttons[self.button_filter]
        if np.any(filtered_buttons):
            rospy.logwarn(f'joystick buttons {np.argwhere(filtered_buttons).tolist()} pressed')
            rospy.logwarn(self.cancel_msg)
            self.giskard.cancel_all_goals()


rospy.init_node('giskard_e_stop')

joy_msg: Joy = rospy.wait_for_message('/joy', Joy)
num_buttons = len(joy_msg.buttons)

button_ids = rospy.get_param('~button_ids', default=list(range(num_buttons)))
muh = MUH(button_ids)
logging.loginfo('giskard joystick e stop is running')
rospy.spin()
