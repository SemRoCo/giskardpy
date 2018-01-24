#!/usr/bin/env python
import numpy as np
import actionlib
import rospy
from collections import defaultdict, OrderedDict
import pylab as plt
from actionlib.simple_action_client import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal, \
    JointTrajectoryControllerState, FollowJointTrajectoryResult
from geometry_msgs.msg._Point import Point
from geometry_msgs.msg._PoseStamped import PoseStamped
from geometry_msgs.msg._Quaternion import Quaternion
from giskard_msgs.msg._Controller import Controller
from giskard_msgs.msg._ControllerListAction import ControllerListAction
from giskard_msgs.msg._ControllerListGoal import ControllerListGoal
from giskard_msgs.msg._ControllerListResult import ControllerListResult
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from sensor_msgs.msg._JointState import JointState
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg._InteractiveMarker import InteractiveMarker
from visualization_msgs.msg._InteractiveMarkerControl import InteractiveMarkerControl
from visualization_msgs.msg._InteractiveMarkerFeedback import InteractiveMarkerFeedback
from visualization_msgs.msg._Marker import Marker

from giskardpy.cartesian_controller import CartesianController
from giskardpy.cartesian_controller_old import CartesianControllerOld
from giskardpy.cartesian_line_controller import CartesianLineController
from giskardpy.donbot import DonBot
from giskardpy.joint_space_control import JointSpaceControl
from giskardpy.pr2 import PR2
from giskardpy.robot import Robot



class RosController(object):
    MAX_TRAJECTORY_TIME = 10

    def __init__(self, robot, cmd_topic, mode=1):
        self.mode = mode

        # tf
        self.tfBuffer = Buffer(rospy.Duration(1))
        self.tf_listener = TransformListener(self.tfBuffer)

        # action server
        self._action_name = 'qp_controller/command'
        self.robot = robot
        self.joint_controller = JointSpaceControl(self.robot)
        #TODO set default joint goal
        # self.cartesian_controller = CartesianController(self.robot)
        self.cartesian_controller = CartesianLineController(self.robot)
        # self.cartesian_controller = CartesianControllerOld(self.robot)
        self.set_default_goals()
        if self.mode == 0:
            self.cmd_pub = rospy.Publisher(cmd_topic, JointState, queue_size=100)
        else:
            self._ac = actionlib.SimpleActionClient('/whole_body_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
            # self._ac = actionlib.SimpleActionClient('/follow_joint_trajectory', FollowJointTrajectoryAction)
            self.state_sub = rospy.Subscriber('/whole_body_controller/state', JointTrajectoryControllerState, self.state_cb)
            # self.state_sub = rospy.Subscriber('/fake_state', JointTrajectoryControllerState, self.state_cb)
        self._as = actionlib.SimpleActionServer(self._action_name, ControllerListAction,
                                                execute_cb=self.action_server_cb, auto_start=False)
        self._as.start()
        self.frequency = 100
        self.rate = rospy.Rate(self.frequency)

        # interactive maker server
        # self.interactive_marker_server = InteractiveMarkerGoal(robot.root_link, robot.end_effectors)
        print('running')

    def state_cb(self, data):
        self.state = data

    def set_default_goals(self):
        self.cartesian_controller.set_goal(self.robot.get_eef_position2())

    def transformPose(self, target_frame, pose, time=None):
        transform = self.tfBuffer.lookup_transform(target_frame,
                                                   pose.header.frame_id,
                                                   pose.header.stamp if time is not None else rospy.Time(0),
                                                   rospy.Duration(1.0))
        new_pose = do_transform_pose(pose, transform)
        return new_pose

    def action_server_cb(self, goal):
        rospy.loginfo('got request')
        success = False
        if goal.type != ControllerListGoal.STANDARD_CONTROLLER:
            rospy.logerr('only standard controller supported')
        else:
            controller = goal.controllers[0]
            if controller.type == Controller.JOINT:
                rospy.loginfo('set joint goal')
                self.joint_controller.set_goal(self.robot.joint_state_msg_to_dict(controller.goal_state))
                c = self.joint_controller
            elif controller.type == Controller.TRANSLATION_3D:
                rospy.loginfo('set cartesian goal')
                root_link_goal = self.transformPose('base_footprint', controller.goal_pose)
                self.cartesian_controller.set_goal({controller.tip_link: self.pose_stamped_to_list(root_link_goal)})
                # print('goal: {}'.format(root_link_goal))
                c = self.cartesian_controller
            # print(self.robot.get_eef_position2())

            #move to goal
            if self.mode == 0:
                muh = defaultdict(list)

                for i in range(self.MAX_TRAJECTORY_TIME):
                    if self._as.is_preempt_requested():
                        rospy.loginfo('new goal, cancel old one')
                        # self._as.set_aborted(ControllerListResult())
                        self._as.set_preempted(ControllerListResult())
                        return
                    # if not self._as.is_preempt_requested():
                    cmd_dict = c.get_next_command()
                    # print(cmd_dict)
                    for k, v in cmd_dict.iteritems():
                        muh[k].append(v)
                    cmd_msg = self.robot.joint_vel_dict_to_msg(cmd_dict)
                    self.cmd_pub.publish(cmd_msg)

                    self.rate.sleep()
            else:
                success = self.create_trajectory(c)


        if success:
            self._as.set_succeeded(ControllerListResult())
            print('success')
        else:
            self._as.set_aborted(ControllerListResult())

    def pose_stamped_to_list(self, pose_stamped):
        return [pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w,
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z, ]

    def create_trajectory(self, controller):
        self._ac.cancel_all_goals()
        goal = FollowJointTrajectoryGoal()
        # goal.trajectory.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
        goal.trajectory.joint_names = self.state.joint_names
        simulated_js = OrderedDict()
        current_js = self.robot.get_joint_state()
        for j in goal.trajectory.joint_names:
            if j in current_js:
                simulated_js[j] = current_js[j]
            else:
                simulated_js[j] = 0
        step_size = 1. / self.frequency
        for k in range(int(self.MAX_TRAJECTORY_TIME/step_size)):
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration((k+1) * step_size)
            if k != 0:
                cmd_dict = controller.get_next_command(simulated_js)
            for i, j in enumerate(goal.trajectory.joint_names):
                if k > 0 and j in cmd_dict:
                    simulated_js[j] += cmd_dict[j]*step_size
                    p.velocities.append(cmd_dict[j])
                else:
                    p.velocities.append(0)
                    pass
                p.positions.append(simulated_js[j])
            goal.trajectory.points.append(p)
            if k > 0 and np.abs(cmd_dict.values()).max() < 0.0025:
                print('done')
                break
        if self._as.is_preempt_requested():
            rospy.loginfo('new goal, cancel old one')
            # self._as.set_aborted(ControllerListResult())
            # self._as.set_preempted(ControllerListResult())
            self._ac.cancel_all_goals()
            return False
        else:
            print('waiting for {:.3f} sec with {} points'.format(p.time_from_start.to_sec(), len(goal.trajectory.points)))
            # r = self._ac.send_goal_and_wait(goal, rospy.Duration(10))
            self.plot_trajectory(goal.trajectory)
            self._ac.send_goal(goal)
            t = rospy.get_rostime()
            while not self._ac.wait_for_result(rospy.Duration(.1)):
                # print('not done yet')
                if self._as.is_preempt_requested():
                    rospy.loginfo('new goal, cancel old one')
                    # self._as.set_aborted(ControllerListResult())
                    # self._as.set_preempted(ControllerListResult())
                    self._ac.cancel_all_goals()
                    return False
            print('shit took {:.3f}s'.format((rospy.get_rostime()-t).to_sec()))
            # if self._ac.wait_for_result(rospy.Duration(10)):
            r = self._ac.get_result()
            print('real result {}'.format(r))
            if r.error_code == FollowJointTrajectoryResult.SUCCESSFUL:
                return True
        return False

    def plot_trajectory(self, tj):
        positions = []
        velocities = []
        for point in tj.points:
            positions.append(point.positions)
            velocities.append(point.velocities)
        positions = np.array(positions)
        velocities = np.array(velocities)
        # plt.plot(positions-positions.mean(axis=0))
        # plt.show()
        # plt.plot(velocities)
        # plt.show()
        pass

if __name__ == '__main__':
    rospy.init_node('giskardpy_controller')

    robot_description = rospy.get_param('robot_description')
    # r = PR2(urdf_str=robot_description)
    # ros_controller = RosController(r, '/pr2/commands')
    r = DonBot(urdf_str=robot_description)
    ros_controller = RosController(r, '/donbot/commands')
    rospy.spin()
