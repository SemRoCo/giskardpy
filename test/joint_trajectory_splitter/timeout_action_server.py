#! /usr/bin/env python

import rospy
import actionlib
import control_msgs.msg


class TimeoutActionServer(object):


    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self._as.register_preempt_callback(self.preempt_requested)

    def preempt_requested(self):
        print('cancel called')
        self._as.set_preempted()

    def execute_cb(self, goal):
        rospy.sleep(10)
        if self._as.is_active():
            self._as.set_succeeded()


if __name__ == '__main__':
    rospy.init_node('TimeooutActionServer')
    name = rospy.get_name()
    server = TimeoutActionServer(name)
    rospy.spin()