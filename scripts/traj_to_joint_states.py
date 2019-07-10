#!/usr/bin/env python
import rospy
from control_msgs.msg import FollowJointTrajectoryActionGoal
from sensor_msgs.msg import JointState


class TrajToJS(object):
    def __init__(self):
        self.traj_sub = rospy.Subscriber('/whole_body_controller/follow_joint_trajectory/goal',
                                         FollowJointTrajectoryActionGoal, self.cb, queue_size=10)
        self.joint_state_pub = rospy.Publisher('/giskard/traj/joint_states', JointState, queue_size=1)

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
        dt = traj.trajectory.points[1].time_from_start.to_sec() - traj.trajectory.points[0].time_from_start.to_sec()
        r = rospy.Rate(1/dt)
        for traj_point in traj.trajectory.points:
            js.header.stamp = start_time + traj_point.time_from_start
            js.position = traj_point.positions
            js.velocity = traj_point.velocities
            self.joint_state_pub.publish(js)
            r.sleep()
        js.velocity = [0 for _ in js.velocity]
        self.joint_state_pub.publish(js)

if __name__ == u'__main__':
    rospy.init_node(u'traj_to_js_publisher')
    traj2js = TrajToJS()
    rospy.spin()
