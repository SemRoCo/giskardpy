#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Vector3
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from giskardpy.python_interface import GiskardWrapper
from giskardpy import tfwrapper
from giskard_msgs.msg import MoveActionGoal, MoveActionResult


class VSMarker():

    def __init__(self):
        self.menu_handler = MenuHandler()
        self.giskard_wrapper = GiskardWrapper()
        self.goal_active = False
        self.marker_pose_publisher = rospy.Publisher("/visual_servoing/goal_update", PoseStamped, queue_size=5)
        self.server = InteractiveMarkerServer("visual_servoing_test_marker")
        self.init_menu()
        self.visual_servoing_test_marker = self.make_marker()
        self.server.insert(self.visual_servoing_test_marker, self.marker_moved_cb)
        self.menu_handler.apply(self.server, self.visual_servoing_test_marker.name)
        self.server.applyChanges()
        self.goal_sub = rospy.Subscriber("/giskard/command/goal", MoveActionGoal, self.giskard_goal_cb)
        self.result_sub = rospy.Subscriber("/giskard/command/result", MoveActionResult, self.giskard_result_cb)

    def giskard_goal_cb(self, msg):
        self.goal_active = True

    def giskard_result_cb(self, msg):
        self.goal_active = False

    def init_menu(self):
        self.menu_handler.insert("Send_VS_Goal", callback=self.send_vs_goal_cb)

    def make_marker(self):
        pose = tfwrapper.lookup_pose('map', 'hand_palm_link')

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.scale = 0.25
        int_marker.pose = pose.pose
        int_marker.name = "visual_servoing_test_marker"

        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        control.always_visible = True

        control.markers.append(marker)
        int_marker.controls.append(control)

        # add visible axes
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        return int_marker

    def marker_moved_cb(self, feedback):
        if self.goal_active:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose = feedback.pose
            self.marker_pose_publisher.publish(goal_pose)

    def send_vs_goal_cb(self, feedback):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose = feedback.pose
        self.giskard_wrapper.set_visual_servoing_goal("odom", "hand_palm_link", goal_pose)
        self.giskard_wrapper.plan_and_execute(wait=False)


if __name__ == '__main__':
    rospy.init_node("visual_servoing_test_marker")
    if tfwrapper.tfBuffer is None:
        tfwrapper.init()

    vs_marker = VSMarker()

    # spin node
    rospy.spin()
