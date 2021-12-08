#! /usr/bin/env python

import rospy
import actionlib
import control_msgs.msg


class FailingActionServer(object):
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
        rospy.sleep(goal.trajectory.points[1].time_from_start)
        result = control_msgs.msg.FollowJointTrajectoryResult()
        result.error_code = control_msgs.msg.FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED
        if self._as.is_active():
            self._as.set_aborted(result)

class FailingActionServer(object):

    def __init__(self):
        self.name_space = rospy.get_param('~name_space')
        self.joint_names = rospy.get_param('~joint_names')
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
        print('cancel called')
        self._as.set_preempted()

    def execute_cb(self, goal):
        rospy.sleep(goal.trajectory.points[1].time_from_start)
        result = control_msgs.msg.FollowJointTrajectoryResult()
        result.error_code = control_msgs.msg.FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED
        if self._as.is_active():
            self._as.set_aborted(result)


if __name__ == '__main__':
    rospy.init_node('SuccessfulActionServer')
    name = rospy.get_name()
    server = FailingActionServer(name)
    rospy.spin()