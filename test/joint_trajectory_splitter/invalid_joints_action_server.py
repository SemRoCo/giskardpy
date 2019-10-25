#! /usr/bin/env python

import rospy
import actionlib
import control_msgs.msg


class SuccessfulAction(object):


    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def execute_cb(self, goal):
        rospy.sleep(goal.trajectory.points[-1].time_from_start)
        result = control_msgs.msg.FollowJointTrajectoryResult()
        result.error_code = control_msgs.msg.FollowJointTrajectoryResult.INVALID_JOINTS
        self._as.set_succeeded(result)


if __name__ == '__main__':
    rospy.init_node('SuccessfulActionServer')
    name = rospy.get_name()
    server = SuccessfulAction(name)
    rospy.spin()