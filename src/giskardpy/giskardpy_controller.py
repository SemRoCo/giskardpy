#!/usr/bin/env python
import numpy as np
import actionlib
import rospy
from collections import defaultdict

from actionlib.simple_action_client import SimpleActionClient
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
from visualization_msgs.msg._InteractiveMarker import InteractiveMarker
from visualization_msgs.msg._InteractiveMarkerControl import InteractiveMarkerControl
from visualization_msgs.msg._InteractiveMarkerFeedback import InteractiveMarkerFeedback
from visualization_msgs.msg._Marker import Marker

from giskardpy.cartesian_controller import CartesianController
from giskardpy.cartesian_controller_old import CartesianControllerOld
from giskardpy.donbot import DonBot
from giskardpy.joint_space_control import JointSpaceControl


class InteractiveMarkerGoal(object):
    def __init__(self):
        # tf
        self.tfBuffer = Buffer(rospy.Duration(1))
        self.tf_listener = TransformListener(self.tfBuffer)

        # marker server
        self.server = InteractiveMarkerServer("eef_control")
        self.menu_handler = MenuHandler()
        int_marker = self.make6DofMarker(InteractiveMarkerControl.MOVE_ROTATE_3D)

        self.server.insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)

        self.server.applyChanges()

        # giskard goal client
        self.client = SimpleActionClient('move', ControllerListAction)
        self.client.wait_for_server()


    def transformPose(self, target_frame, pose, time=None):
        transform = self.tfBuffer.lookup_transform(target_frame,
                                                   pose.header.frame_id,
                                                   pose.header.stamp if time is not None else rospy.Time(0),
                                                   rospy.Duration(1.0))
        new_pose = do_transform_pose(pose, transform)
        return new_pose

    def makeBox(self, msg):
        marker = Marker()

        marker.type = Marker.SPHERE
        marker.scale.x = msg.scale * 0.2
        marker.scale.y = msg.scale * 0.2
        marker.scale.z = msg.scale * 0.2
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.5

        return marker

    def makeBoxControl(self, msg):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeBox(msg))
        msg.controls.append(control)
        return control

    def make6DofMarker(self, interaction_mode):
        int_marker = InteractiveMarker()

        p = PoseStamped()
        p.header.frame_id = 'gripper_tool_frame'
        p.pose.orientation.w = 1

        int_marker.header.frame_id = "gripper_tool_frame"
        int_marker.pose.orientation.w = 1
        int_marker.pose = self.transformPose('base_footprint', p).pose
        int_marker.header.frame_id = "base_footprint"
        int_marker.scale = .2

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        # insert a box
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D: "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D: "MOVE_ROTATE_3D"}
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 0, 1)
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 1, 0, 1)
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 1, 0, 1)
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 1, 1)
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation = Quaternion(0, 0, 1, 1)
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        return int_marker

    def processFeedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            print('interactive goal update')
            goal = ControllerListGoal()
            goal.type = ControllerListGoal.STANDARD_CONTROLLER
            c = Controller()
            c.type = Controller.TRANSFORM_6D
            c.goal_pose.header = feedback.header
            c.goal_pose.pose = feedback.pose

            c.tip_link = 'gripper_tool_frame'
            goal.controllers.append(c)
            # self.client.cancel_all_goals()
            self.client.send_goal(goal)
        self.server.applyChanges()


class RosController(object):
    MAX_ITERATIONS = 10000

    def __init__(self, robot):
        # tf
        self.tfBuffer = Buffer(rospy.Duration(1))
        self.tf_listener = TransformListener(self.tfBuffer)

        # action server
        self._action_name = 'move'
        self.robot = r
        self.joint_controller = JointSpaceControl(self.robot)
        # self.cartesian_controller = CartesianController(self.robot)
        self.cartesian_controller = CartesianControllerOld(self.robot)
        self.cmd_pub = rospy.Publisher('/donbot/commands', JointState, queue_size=100)
        self._as = actionlib.SimpleActionServer(self._action_name, ControllerListAction,
                                                execute_cb=self.action_server_cb, auto_start=False)
        self._as.start()
        frequency = 100
        self.rate = rospy.Rate(frequency)
        print('running')

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
            elif controller.type == Controller.TRANSFORM_6D:
                rospy.loginfo('set cartesian goal')
                root_link_goal = self.transformPose('base_footprint', controller.goal_pose)
                self.cartesian_controller.set_goal({controller.tip_link: self.pose_stamped_to_list(root_link_goal)})
                c = self.cartesian_controller
            # print(self.robot.get_eef_position2())

            #move to goal
            muh = defaultdict(list)

            for i in range(self.MAX_ITERATIONS):
                if self._as.is_preempt_requested():
                    rospy.loginfo('new goal, cancel old one')
                    # self._as.set_aborted(ControllerListResult())
                    self._as.set_preempted(ControllerListResult())
                    break
                # if not self._as.is_preempt_requested():
                cmd_dict = c.get_next_command()
                # print(cmd_dict)
                for k, v in cmd_dict.iteritems():
                    muh[k].append(v)
                cmd_msg = self.robot.joint_vel_dict_to_msg(cmd_dict)
                err = np.linalg.norm(cmd_msg.velocity)
                if err < 0.005:
                    rospy.loginfo('goal reached')
                    success = True
                    break
                self.cmd_pub.publish(cmd_msg)

                self.rate.sleep()
            # for k, v in muh.items():
            #     print('{}: avg {}, max {}'.format(k, np.mean(muh[k]), np.max(np.abs(muh[k]))))


        if success:
            self._as.set_succeeded(ControllerListResult())
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

    def loop(self):
        pass


if __name__ == '__main__':
    rospy.init_node('giskardpy_controller')

    r = DonBot(1, '/home/stelter/giskardpy_ws/src/iai_robots/iai_donbot_description/robots/iai_donbot.urdf')
    ros_controller = RosController(r)
    interactive_marker_goal = InteractiveMarkerGoal()
    rospy.spin()
