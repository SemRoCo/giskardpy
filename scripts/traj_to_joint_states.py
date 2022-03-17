#!/usr/bin/env python
import rospy
from control_msgs.msg import FollowJointTrajectoryActionGoal
from sensor_msgs.msg import JointState
import giskardpy.utils.tfwrapper as tf
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import msg_to_kdl
import PyKDL as kdl

class TrajToJS(object):
    def __init__(self, odom_x_joint, odom_y_joint, odom_z_joint, odom_frame):
        self.traj_sub = rospy.Subscriber('/whole_body_controller/follow_joint_trajectory/goal',
                                         FollowJointTrajectoryActionGoal, self.cb, queue_size=10)
        self.joint_state_pub = rospy.Publisher('/giskard/traj/joint_states', JointState, queue_size=10)
        self.odom_x = odom_x_joint
        self.odom_y = odom_y_joint
        self.odom_z = odom_z_joint
        self.odom_frame = odom_frame
        tf.init()


    def cb(self, traj):
        """
        :type traj: FollowJointTrajectoryActionGoal
        :return:
        """
        map_T_odom_original = msg_to_kdl(tf.lookup_pose('map', self.odom_frame))

        traj = traj.goal
        start_time = traj.trajectory.header.stamp
        js = JointState()
        js.name = traj.trajectory.joint_names
        odom_x_id = js.name.index(self.odom_x)
        odom_y_id = js.name.index(self.odom_y)
        odom_z_id = js.name.index(self.odom_z)
        rospy.sleep(start_time - rospy.get_rostime())
        dt = traj.trajectory.points[1].time_from_start.to_sec() - traj.trajectory.points[0].time_from_start.to_sec()
        r = rospy.Rate(1/dt)
        for traj_point in traj.trajectory.points:
            odom_T_map = msg_to_kdl(tf.lookup_pose(self.odom_frame, 'map'))
            js.header.stamp = start_time + traj_point.time_from_start
            js.position = list(traj_point.positions)
            js.velocity = traj_point.velocities
            odom_x = js.position[odom_x_id]
            odom_y = js.position[odom_y_id]
            odom_z = js.position[odom_z_id]
            odom_original_T_goal = kdl.Frame(kdl.Rotation().RotZ(odom_z), kdl.Vector(odom_x, odom_y, 0))
            map_T_goal = map_T_odom_original * odom_original_T_goal
            odom_T_goal = odom_T_map * map_T_goal
            js.position[odom_x_id] = odom_T_goal.p[0]
            js.position[odom_y_id] = odom_T_goal.p[1]
            js.position[odom_z_id] = kdl.Rotation().RotZ(odom_z).GetRot()[2]
            self.joint_state_pub.publish(js)
            r.sleep()
        js.velocity = [0 for _ in js.velocity]
        self.joint_state_pub.publish(js)

if __name__ == '__main__':
    rospy.init_node('traj_to_js_publisher')
    try:
        traj2js = TrajToJS(odom_x_joint=rospy.get_param('~odom_x_joint'),
                           odom_y_joint=rospy.get_param('~odom_y_joint'),
                           odom_z_joint=rospy.get_param('~odom_z_joint'),
                           odom_frame=rospy.get_param('~odom_frame'))
        rospy.sleep(0.5)
        logging.loginfo('running')
        rospy.spin()
    except KeyError:
        logging.loginfo(
            'Example call: rosrun giskardpy traj_to_joint_states.py _odom_x_joint:=odom_x_joint _odom_y_joint:=odom_y_joint _odom_z_joint:=odom_z_joint _odom_frame:=odom')

