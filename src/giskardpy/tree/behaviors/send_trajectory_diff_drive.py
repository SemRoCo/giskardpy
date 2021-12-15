from threading import Thread

import rospy
from geometry_msgs.msg import Twist
from py_trees import Status
from trajectory_msgs.msg import JointTrajectoryPoint

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging


class SendFollowJointTrajectoryDiffDrive(GiskardBehavior):
    def __init__(self, name, namespace):
        super().__init__(name)
        self.namespace = namespace
        self.cmd_vel_topic = '{}/cmd_vel'.format(self.namespace)
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.world.register_controlled_joints(['diff_drive'])

    def initialise(self):
        super(SendFollowJointTrajectoryDiffDrive, self).initialise()
        self.trajectory = self.get_god_map().get_data(identifier.trajectory)
        # self.trajectory = self.trajectory.to_msg(0.05, ['diff_drive/trans', 'diff_drive/rot'], True)
        self.trajectory = self.trajectory.to_msg(0.05, ['diff_drive/r_wheel', 'diff_drive/l_wheel'], True)
        self.update_thread = Thread(target=self.worker)
        self.update_thread.start()

    def update(self):
        if self.update_thread.is_alive():
            return Status.RUNNING
        return Status.SUCCESS

    def worker(self):
        start_time = self.trajectory.header.stamp
        cmd = Twist()
        rospy.sleep(start_time - rospy.get_rostime())
        dt = self.trajectory.points[1].time_from_start.to_sec() - self.trajectory.points[0].time_from_start.to_sec()
        r = rospy.Rate(1/dt)
        for traj_point in self.trajectory.points:  # type: JointTrajectoryPoint
            wheel_dist = 0.404
            wheel_radius = 0.098
            r_wheel_vel = traj_point.velocities[0]
            l_wheel_vel = traj_point.velocities[1]
            rot_vel = wheel_radius / wheel_dist * (r_wheel_vel - l_wheel_vel)
            trans_vel = wheel_radius / 2 * (r_wheel_vel + l_wheel_vel)
            cmd.linear.x = trans_vel
            cmd.angular.z = rot_vel
            self.vel_pub.publish(cmd)
            r.sleep()
        self.vel_pub.publish(Twist())

    def terminate(self, new_status):
        try:
            self.update_thread.join()
        except Exception as e:
            # FIXME sometimes terminate gets called without init being called
            # happens when a previous plugin fails
            logging.logwarn('terminate was called before init')
        super().terminate(new_status)
