import abc
from abc import ABC
from threading import Thread
from typing import List

import rospy
import rostopic
from geometry_msgs.msg import Twist
from py_trees import Status
from rospy import ROSException
from rostopic import ROSTopicException

import giskardpy.identifier as identifier
from giskardpy.goals.base_traj_follower import BaseTrajFollower
from giskardpy.goals.goal import Goal
from giskardpy.model.joints import OmniDrive
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import catch_and_raise_to_blackboard


class SendTrajectoryClosedLoop(GiskardBehavior, ABC):
    def __init__(self, name, **kwargs):
        super().__init__(name)

    def setup(self, timeout):
        super().setup(timeout)
        self.put_drive_goals_on_godmap()

    def put_drive_goals_on_godmap(self):
        try:
            drive_goals = self.god_map.get_data(identifier.drive_goals)
        except KeyError:
            drive_goals = []
        drive_goals.extend(self.get_drive_goals())
        self.god_map.set_data(identifier.drive_goals, drive_goals)

    @abc.abstractmethod
    def get_drive_goals(self) -> List[Goal]:
        """
        """

    @abc.abstractmethod
    def update(self):
        pass


class OmniDriveCmdVel(SendTrajectoryClosedLoop):
    min_deadline: rospy.Time
    max_deadline: rospy.Time
    update_thread: Thread
    supported_state_types = [Twist]

    @profile
    def __init__(self, name, cmd_vel_topic, goal_time_tolerance=1):
        super().__init__(name)
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
                self.joint = joint
        self.world.register_controlled_joints(self.controlled_joints)
        loginfo(f'Received controlled joints from \'{cmd_vel_topic}\'.')

    def get_drive_goals(self) -> List[Goal]:
        return [BaseTrajFollower(god_map=self.god_map, joint_name=self.joint.name)]

    @profile
    @catch_and_raise_to_blackboard
    def initialise(self):
        super().initialise()
        self.trajectory = self.get_god_map().get_data(identifier.trajectory)
        sample_period = self.god_map.unsafe_get_data(identifier.sample_period)
        self.start_time = self.god_map.unsafe_get_data(identifier.tracking_start_time)
        self.trajectory = self.trajectory.to_msg(sample_period, self.start_time, [self.joint], True)
        self.end_time = self.start_time + self.trajectory.points[-1].time_from_start
        # self.update_thread = Thread(target=self.worker)
        # self.update_thread.start()

    @catch_and_raise_to_blackboard
    def update(self):
        t = rospy.get_rostime()
        if self.start_time > t:
            self.vel_pub.publish(Twist())
            return Status.RUNNING
        if t <= self.end_time:
            cmd = self.god_map.get_data(identifier.qp_solver_solution)
            twist = Twist()
            try:
                twist.linear.x = cmd[0][self.joint.x_vel.position_name]
            except:
                twist.linear.x = 0
            try:
                twist.linear.y = cmd[0][self.joint.y_vel.position_name]
            except:
                twist.linear.y = 0
            try:
                twist.angular.z = cmd[0][self.joint.rot_vel.position_name]
            except:
                twist.angular.z = 0
            # print(f'twist: {twist.linear.x:.4} {twist.linear.y:.4} {twist.angular.z:.4}')
            # print_dict(self.god_map.get_data(identifier.debug_expressions_evaluated))
            # print('-----------------')
            self.vel_pub.publish(twist)
            return Status.RUNNING
        self.vel_pub.publish(Twist())
        return Status.SUCCESS
        # if self.update_thread.is_alive():
        #     return Status.RUNNING
        # return Status.SUCCESS

    # def worker(self):
    #     start_time = self.trajectory.header.stamp
    #     end_time = start_time + self.trajectory.points[-1].time_from_start
    #     twist = Twist()
    #     # rospy.sleep(start_time - rospy.get_rostime())
    #     # dt = self.trajectory.points[1].time_from_start.to_sec() - self.trajectory.points[0].time_from_start.to_sec()
    #     r = rospy.Rate(100)
    #     cmd = self.god_map.get_data(identifier.qp_solver_solution)
    #     # for traj_point in self.trajectory.points:  # type: JointTrajectoryPoint
    #     # time = rospy.get_rostime()
    #     while rospy.get_rostime() < end_time:
    #         # base_footprint_T_odom = self.world.get_fk(self.joint.child_link_name, self.joint.parent_link_name)
    #         # translation_velocity = np.array([traj_point.velocities[0], traj_point.velocities[1], 0, 0])
    #         # translation_velocity = np.dot(base_footprint_T_odom, translation_velocity)
    #         twist.linear.x = cmd[0][self.joint.x_vel.position_name]
    #         twist.linear.y = cmd[0][self.joint.y_vel.position_name]
    #         twist.angular.z = cmd[0][self.joint.rot_vel.position_name]
    #         # next_time = rospy.get_rostime()
    #         # self.joint.update_state(cmd, (next_time - time).to_sec())
    #         # time = next_time
    #         print(f'twist: {twist.linear.x:.4} {twist.linear.y:.4} {twist.angular.z:.4}')
    #         # print(f'state: {self.world.state[self.joint.x_vel_name].velocity} '
    #         #       f'{self.world.state[self.joint.y_vel_name].velocity} '
    #         #       f'{self.world.state[self.joint.rot_vel_name].velocity}')
    #         print_dict(self.god_map.get_data(identifier.debug_expressions_evaluated))
    #         print('-----------------')
    #         self.vel_pub.publish(twist)
    #         r.sleep()
    #     self.vel_pub.publish(Twist())

    # def terminate(self, new_status):
    #     try:
    #         self.update_thread.join()
    #     except Exception as e:
    #         # FIXME sometimes terminate gets called without init being called
    #         # happens when a previous plugin fails
    #         logging.logwarn('terminate was called before init')
    #     super().terminate(new_status)
