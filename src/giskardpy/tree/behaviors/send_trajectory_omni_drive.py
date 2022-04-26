from threading import Thread
import control_msgs
from geometry_msgs.msg import Twist
from py_trees import Status
from rospy import ROSException
from rostopic import ROSTopicException
from sensor_msgs.msg import JointState

from giskardpy.model.joints import OmniDrive

import rospy
import rostopic
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo


class SendTrajectoryOmniDrive(GiskardBehavior):
    min_deadline: rospy.Time
    max_deadline: rospy.Time
    supported_state_types = [Twist]

    @profile
    def __init__(self, name, cmd_vel_topic, drive, goal_time_tolerance=1, fill_velocity_values=True):
        GiskardBehavior.__init__(self, name)
        self.fill_velocity_values = fill_velocity_values
        self.goal_time_tolerance = rospy.Duration(goal_time_tolerance)

        loginfo(f'Waiting for cmd_vel topic \'{cmd_vel_topic}\' to appear.')
        try:
            msg_type, _, _ = rostopic.get_topic_class(cmd_vel_topic)
            if msg_type is None:
                raise ROSTopicException()
            if msg_type not in self.supported_state_types:
                raise TypeError(f'Cmd_vel topic of type \'{msg_type}\' is not supported. '
                                f'Must be one of: \'{self.supported_state_types}\'')
            self.vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        except ROSException as e:
            logging.logwarn(f'Couldn\'t connect to {cmd_vel_topic}. Is it running?')
            rospy.sleep(1)

        for joint in self.world.joints.values():
            if isinstance(joint, OmniDrive):
                # FIXME can only handle one drive
                self.controlled_joints = [joint]
        self.world.register_controlled_joints(self.controlled_joints)
        loginfo('Received controlled joints from \'{}\'.'.format(cmd_vel_topic))

    @profile
    def initialise(self):
        super().initialise()
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
        r = rospy.Rate(1 / dt)
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
