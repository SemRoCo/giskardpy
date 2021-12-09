#! /usr/bin/env python

import actionlib
import control_msgs.msg
import rospy
from control_msgs.msg import FollowJointTrajectoryResult, FollowJointTrajectoryGoal

from giskardpy.utils import logging


class FakeActionServer(object):

    def __init__(self):
        self.name_space = rospy.get_param('~name_space')
        self.joint_names = rospy.get_param('~joint_names')
        self.sleep_percent = rospy.get_param('~sleep_factor')
        self.result = rospy.get_param('~result')
        self.state = {j:0 for j in self.joint_names}
        self.pub = rospy.Publisher('{}/state'.format(self.name_space), control_msgs.msg.JointTrajectoryControllerState,
                                   queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.state_cb)
        self._as = actionlib.SimpleActionServer(self.name_space, control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self._as.register_preempt_callback(self.preempt_requested)

    def state_cb(self, timer_event):
        msg = control_msgs.msg.JointTrajectoryControllerState()
        msg.header.stamp = timer_event.current_real
        msg.joint_names = self.joint_names
        self.pub.publish(msg)

    def preempt_requested(self):
        logging.loginfo('cancel called')
        self._as.set_preempted()

    def execute_cb(self, goal: FollowJointTrajectoryGoal):
        wait_until = goal.trajectory.header.stamp + self.sleep_percent * goal.trajectory.points[-1].time_from_start
        while rospy.get_rostime() < wait_until:
            rospy.sleep(0.1)
        if self._as.is_active():
            result = control_msgs.msg.FollowJointTrajectoryResult()
            result.error_code = self.result
            if self.result == FollowJointTrajectoryResult.SUCCESSFUL:
                self._as.set_succeeded(result)
            else:
                self._as.set_aborted(result)


if __name__ == '__main__':
    rospy.init_node('SuccessfulActionServer')
    server = FakeActionServer()
    rospy.spin()
