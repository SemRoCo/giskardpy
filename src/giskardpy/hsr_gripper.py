#!/usr/bin/env python
# Copyright (C) 2016 Toyota Motor Corporation
import actionlib
import control_msgs.msg
import controller_manager_msgs.srv
import rospy
import trajectory_msgs.msg
from listener import Listener
from sensor_msgs.msg import JointState


# rospy.init_node('test')

class HsrGripper:
    """
    This class controls and directs the controller of arm
    """

    _running = False

    def __init__(self):
        # initialize action client
        self.cli = actionlib.SimpleActionClient(
            '/hsrb/gripper_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)

        # wait for the action server to establish connection
        self.cli.wait_for_server()

        # make sure the controller is running
        rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
        list_controllers = rospy.ServiceProxy(
            '/hsrb/controller_manager/list_controllers',
            controller_manager_msgs.srv.ListControllers)
        self._running = False
        while self._running is False:
            rospy.sleep(0.1)
            for c in list_controllers().controller:
                if c.name == 'gripper_controller' and c.state == 'running':
                    self._running = True

    def move_gripper(self, position_value, velocity, effort):
        """
        this method is used to catch objects with the grippers
        :param position_value: float
        :param velocity: float
        :param effort: float
        :return:
        """
        # fill ROS message
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = ["hand_motor_joint"]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [position_value - 0.42]
        p.velocities = [velocity]
        p.effort = [effort]
        p.time_from_start = rospy.Time(3)
        traj.points = [p]
        goal.trajectory = traj

        # send message to the action server
        self.cli.send_goal(goal)

        # wait for the action server to complete the order
        self.cli.wait_for_result()
        return self._running

    def close_gripper(self):
        """
        this method close the gripper
        :return:
        """
        return self.move_gripper(0.05, 1, 1)

    def open_gripper(self):
        """
        this method open the gripper.
        :return:
        """
        return self.move_gripper(1.2, 1, 1)

    def object_in_gripper(self, width_object):
        """
        This method checks if the object is in the gripper, then position_value of gripper should be greater as -0.5
        :param width_object: float
        :return: false or true, boolean
        """
        self.move_gripper(-2, 1, 0.8)
        l = Listener()
        l.set_topic_and_typMEssage("/hsrb/joint_states", JointState)
        l.listen_topic_with_sensor_msg()
        current_hand_motor_value = l.get_value_from_sensor_msg("hand_motor_joint")
        print("Current hand motor joint is:")
        print current_hand_motor_value
        print("Is object in gripper ?")
        print current_hand_motor_value >= -0.5
        return current_hand_motor_value >= -0.5


if __name__ == '__main__':
    rospy.init_node('check_gripper')
    gripper = HsrGripper()
    gripper.move_gripper(-0.05, 1, 0.8)