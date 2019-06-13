#!/usr/bin/env python
import rospy
from control_msgs.msg import FollowJointTrajectoryActionGoal
from sensor_msgs.msg import JointState


class TrajToJS(object):
    def __init__(self):
        self.traj_sub = rospy.Subscriber('/whole_body_controller/follow_joint_trajectory/goal',
                                         FollowJointTrajectoryActionGoal, self.cb, queue_size=10)
        self.joint_state_pub = rospy.Publisher('/giskard/joint_states', JointState, queue_size=1)

    def cb(self, traj):
        """
        :type traj: FollowJointTrajectoryActionGoal
        :return:
        """
        traj = traj.goal
        start_time = traj.trajectory.header.stamp
        js = JointState()
        js.name = traj.trajectory.joint_names
        rospy.sleep(start_time - rospy.get_rostime())
        r = rospy.Rate(20)
        for traj_point in traj.trajectory.points:
            js.header.stamp = start_time + traj_point.time_from_start
            js.position = traj_point.positions
            js.velocity = traj_point.velocities
            self.joint_state_pub.publish(js)
            r.sleep()

if __name__ == u'__main__':
    rospy.init_node(u'giskard')
    traj2js = TrajToJS()
    rospy.spin()