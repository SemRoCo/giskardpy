#!/usr/bin/env python

import rospy
import sys
import tf
from giskardpy.python_interface import GiskardWrapper
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from iai_naive_kinematics_sim.srv import SetJointState, SetJointStateRequest, SetJointStateResponse
from sensor_msgs.msg import JointState
import numpy as np
from tf.transformations import quaternion_about_axis
import tf2_ros
import tf2_geometry_msgs
from giskardpy.qp_problem_builder import SoftConstraint
from collections import OrderedDict
from geometry_msgs.msg import Vector3Stamped, Vector3
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message
from giskardpy import symengine_wrappers as sw
from giskardpy import constraints as cnst
from giskardpy import god_map as gm
from giskardpy import tfwrapper as tf_wrapper
from urdf_parser_py.urdf import URDF
import math
import yaml
from giskardpy import utils_constraints as uc
from giskardpy.tfwrapper import lookup_pose, pose_to_kdl, np_to_kdl, kdl_to_pose


class Gripper:

    def __init__(self):
        """
     this class controls and performs mvt of gripper
     """
        self.giskard = GiskardWrapper()
        rospy.logout("--> Set kitchen/world in Giskard")

        rospy.logout("- Set pose kitchen")
        kitchen_pose = tf_wrapper.lookup_pose("map", "iai_kitchen/world")
        kitchen_pose.header.frame_id = "map"

        rospy.logout("- Get urdf")
        self.urdf = rospy.get_param("kitchen_description")
        self.parsed_urdf = URDF.from_xml_string(self.urdf)
        self.config_file = None

        rospy.logout("- clear urdf and add kitchen urdf")
        self.giskard.clear_world()
        self.giskard.add_urdf(name="kitchen", urdf=self.urdf, pose=kitchen_pose, js_topic="kitchen/cram_joint_states")

        rospy.logout("- Set right and left Gripper service proxy")
        self.l_gripper_service = rospy.ServiceProxy('/l_gripper_simulator/set_joint_states', SetJointState)
        self.r_gripper_service = rospy.ServiceProxy('/r_gripper_simulator/set_joint_states', SetJointState)

        rospy.logout("--> Gripper are ready for every task.")

    def some_mvt(self, goal, body):
        """
        execute of simple movement of position and rotation with the grippers
        :param goal: goal name
        :type: str
        :param body: End Link, l_gripper or r_gripper
        :type: str
        :param axis: the axis on which the object must be caught, maybe 'x', 'y' or 'z'
        :type: str
        :return:
        """
        #self.giskard.set_json_goal("MoveWithConstraint", goal_name=goal, body_name=body)
        self.giskard.set_json_goal("FrameConstraint", goal_name=goal, body_name=body)
        #self.giskard.set_json_goal("CartesianPoseUpdate", root_link="odom_combined", tip_link=body, goal_name=goal)
        #self.giskard.avoid_all_collisions()
        self.giskard.allow_all_collisions()
        self.giskard.plan_and_execute()

    def open_or_close_with_translation(self, goal, body, action):
        """
        execute simple translation of gripper with something
        :param goal: goal name (Frame)
        :type: str
        :param body: End Link, l_gripper or r_gripper
        :type: str
        :param action: percentage of movement
        :type: float
        :return:
        """
        #self.giskard.set_json_goal("TranslationalAngularConstraint", goal_name=goal, body_name=body, action=action)
        #self.giskard.set_json_goal("FrameTranslationConstraint", goal_name=goal, body_name=body, action=action)
        self.giskard.set_json_goal("AngularConstraint", goal_name=goal, body_name=body, action=action)
        self.giskard.allow_all_collisions()
        self.giskard.plan_and_execute()

    def do_rotational_mvt(self, goal, body, action):
        """
        Execute rotational movement with the gripper(body)
        :param goal: goal name
        :type: str
        :param body: End Link, l/r Gripper
        :str:  str
        :param action: percentage of movement
        :rtype: float
        :return:
        """
        self.giskard.set_json_goal("RotationalConstraint", goal_name=goal, body_name=body, action=action)
        # self.giskard.allow_collision(robot_links=(body), link_bs=(goal))
        self.giskard.allow_all_collisions()
        self.giskard.plan_and_execute()

    def mvt_l_gripper(self, l_finger_joint, r_finger_joint, l_finger_tip_joint, r_finger_tip_joint):
        """
        execute mvt of left gripper
        :param l_finger_joint: value of l_gripper_l_finger_joint
        :type: float
        :param r_finger_joint: value of l_gripper_r_finger_joint
        :type: float
        :param l_finger_tip_joint: value of l_gripper_l_finger_tip_joint
        :type: float
        :param r_finger_tip_joint: value of l_gripper_r_finger_tip_joint
        :type: float
        :return:
        """
        rospy.logout('Start mvt L gripper')
        gripper_request = SetJointStateRequest()
        gripper_request.state.name = ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
                                      'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_tip_joint']
        gripper_request.state.position = [l_finger_joint, r_finger_joint, l_finger_tip_joint, r_finger_tip_joint]
        gripper_request.state.velocity = [0, 0, 0, 0]
        gripper_request.state.effort = [0, 0, 0, 0]
        self.l_gripper_service.call(gripper_request)
        rospy.logout('End mvt L gripper')

    def mvt_r_gripper(self, l_finger_joint, r_finger_joint, l_finger_tip_joint, r_finger_tip_joint):
        """
        execute mvt of right gripper
        :param l_finger_joint: value of r_gripper_l_finger_joint
        :type: float
        :param r_finger_joint: value of r_gripper_r_finger_joint
        :type: float
        :param l_finger_tip_joint: value of r_gripper_l_finger_tip_joint
        :type: float
        :param r_finger_tip_joint: value of r_gripper_r_finger_tip_joint
        :type: float
        :return:
        """
        rospy.logout('Start mvt R gripper')
        gripper_request = SetJointStateRequest()
        gripper_request.state.name = ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint',
                                      'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_tip_joint']
        gripper_request.state.position = [l_finger_joint, r_finger_joint, l_finger_tip_joint, r_finger_tip_joint]
        gripper_request.state.velocity = [0, 0, 0, 0]
        gripper_request.state.effort = [0, 0, 0, 0]
        self.r_gripper_service.call(gripper_request)
        rospy.logout('End mvt R gripper')

    def get_pose(self, source, frame_id):
        """
        the method get exactly the pose of link frame_id to source(map or odom)
        :param source: odom or map, typ string
        :param frame_id: string
        :return: PoseStamped
        """
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        # print tf_listener
        ps = PoseStamped()

        transform = tf_buffer.lookup_transform(source,
                                               frame_id,
                                               rospy.Time(0),
                                               rospy.Duration(5.0))
        # print transform

        return tf2_geometry_msgs.do_transform_pose(ps, transform)

    def info_about_joints(self, joint_name):
        """
        This method get info about some joint ( limits and axis)
        :param joint_name: name of the joint
        :type: String
        :return: joint_limit, joint_axis
        :rtype: {'effort': float, 'lower': float, 'upper': float, 'velocity': float}, array 1*3
        """
        joint_limits = {}
        joint_axis = None
        for j in self.parsed_urdf.joints:
            if joint_name == j.name:
                if j.axis != None:
                    joint_axis = j.axis
                if j.limit != None:
                    joint_limits = {'effort': j.limit.effort, 'lower': j.limit.lower,
                                    'upper': j.limit.upper, 'velocity': j.limit.velocity}
        return joint_limits, joint_axis

    def set_kitchen_goal(self, joint_name, joint_value):
        self.giskard.set_object_joint_state(object_name='kitchen',
                                            joint_states={joint_name: joint_value})


def reset_config_file(path="."):
    config_file_manager = uc.ConfigFileManager()
    config_file_manager.load_urdf("kitchen_description")
    config_file_manager.set_yaml_config_file("config_file_002.yaml")
    with open(path, "r") as file_reader:
        lines = file_reader.read().splitlines()
        print lines
        for line in lines:
            splited_line = line.split(" ")
            config_file_manager.update_joint_of_config_file(path="config_file_002.yaml", joint_name=splited_line[0],
                                                            constraint_type=splited_line[1],
                                                            controllable_link=splited_line[2],
                                                            grasp_axis=splited_line[3])


def set_cart_goal_test():
    giskard = GiskardWrapper()
    poseStamp = lookup_pose("odom_combined", "iai_kitchen/kitchen_island_left_lower_drawer_handle")
    h_g = np_to_kdl(np.array([[-1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]]))
    pose = pose_to_kdl(poseStamp.pose)

    pose = pose * h_g
    poseStamp.pose = kdl_to_pose(pose)
    giskard.set_cart_goal("odom_combined", "r_gripper_tool_frame", poseStamp)
    giskard.allow_all_collisions()
    giskard.plan_and_execute()


# gripper.do_rotational_mvt('iai_kitchen/sink_area_left_middle_drawer_handle', 'l_gripper_tool_frame', -1)

def open_kitchen_island_object():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt("iai_kitchen/kitchen_island_left_lower_drawer_handle", "r_gripper_tool_frame")
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation("iai_kitchen/kitchen_island_left_lower_drawer_handle",
                                           "r_gripper_tool_frame", 1)
    gripper.set_kitchen_goal("kitchen_island_left_lower_drawer_main_joint", 0.48)

def close_kitchen_island_object():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt("iai_kitchen/kitchen_island_left_lower_drawer_handle", "r_gripper_tool_frame")
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation("iai_kitchen/kitchen_island_left_lower_drawer_handle",
                                           "r_gripper_tool_frame", -1)
    gripper.set_kitchen_goal("kitchen_island_left_lower_drawer_main_joint", 0.0)


def open_oven_door_object():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/oven_area_oven_door_handle', 'l_gripper_tool_frame')
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation('iai_kitchen/oven_area_oven_door_handle', 'l_gripper_tool_frame', -0.5)
    gripper.set_kitchen_goal('oven_area_oven_door_joint', 0.785)

def close_oven_door_object():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/oven_area_oven_door_handle', 'l_gripper_tool_frame')
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation('iai_kitchen/oven_area_oven_door_handle', 'l_gripper_tool_frame', 0.5)
    gripper.set_kitchen_goal('oven_area_oven_door_joint', 0.0)


def close_sink_area_object():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/sink_area_left_middle_drawer_handle', 'l_gripper_tool_frame')
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation('iai_kitchen/sink_area_left_middle_drawer_handle', 'l_gripper_tool_frame',
                                           1)
    gripper.set_kitchen_goal('sink_area_left_middle_drawer_main_joint', 0.00)

def open_sink_area_object():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/sink_area_left_middle_drawer_handle', 'l_gripper_tool_frame')
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation('iai_kitchen/sink_area_left_middle_drawer_handle', 'l_gripper_tool_frame',
                                           -1)
    gripper.set_kitchen_goal('sink_area_left_middle_drawer_main_joint', 0.48)


def open_fridge_door():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/iai_fridge_door_handle', 'r_gripper_tool_frame')
    #gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    #gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation('iai_kitchen/iai_fridge_door_handle', 'r_gripper_tool_frame', 1)
    gripper.set_kitchen_goal('iai_fridge_door_joint', 1.57)

def close_fridge_door():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/iai_fridge_door_handle', 'r_gripper_tool_frame')
    #gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    #gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.open_or_close_with_translation('iai_kitchen/iai_fridge_door_handle', 'r_gripper_tool_frame', -1)
    gripper.set_kitchen_goal('iai_fridge_door_joint', 0)

def rotate_oven_knob():
    gripper = Gripper()
    gripper.mvt_l_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.mvt_r_gripper(0.54, 0.54, 0.54, 0.54)
    gripper.some_mvt('iai_kitchen/oven_area_oven_knob_stove_4', 'l_gripper_tool_frame')
    gripper.mvt_l_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.mvt_r_gripper(0.0, 0.0, 0.0, 0.0)
    gripper.do_rotational_mvt('iai_kitchen/oven_area_oven_knob_stove_4', 'l_gripper_tool_frame', 1)

def reset_ktichen( list_joints):
    gripper = Gripper()
    for j in list_joints:
        gripper.set_kitchen_goal(j, 0.0)

def test_jacobi_fwd(l1, l2, theta1, theta2):
    x1_d = -l1 * theta1 * np.sin(theta1) - l2 * theta1 * np.sin(theta1 + theta2) - l2 * np.sin(theta2 + theta1)
    x2_d = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l2 * np.cos(theta2 + theta1)

    x1 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    x2 = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

    return np.dot(np.array([x1_d, x2_d]), np.array([x2, x1]))

if __name__ == '__main__':
    rospy.init_node('do_test_constraint')
    list_joint = [
        "sink_area_trash_drawer_main_joint",
        "sink_area_left_upper_drawer_main_joint",
        "sink_area_left_middle_drawer_main_joint",
        "sink_area_left_bottom_drawer_main_joint",
        "sink_area_dish_washer_main_joint",
        "sink_area_dish_washer_door_joint",
        "oven_area_oven_door_joint",
        "oven_area_oven_knob_stove_1_joint",
        "oven_area_oven_knob_stove_2_joint",
        "oven_area_oven_knob_stove_3_joint",
        "oven_area_oven_knob_stove_4_joint",
        "oven_area_oven_knob_oven_joint",
        "oven_area_area_middle_upper_drawer_main_joint",
        "oven_area_area_middle_lower_drawer_main_joint",
        "oven_area_area_left_drawer_main_joint",
        "oven_area_area_right_drawer_main_joint",
        "kitchen_island_left_upper_drawer_main_joint",
        "kitchen_island_left_lower_drawer_main_joint",
        "kitchen_island_middle_upper_drawer_main_joint",
        "kitchen_island_middle_lower_drawer_main_joint",
        "kitchen_island_right_upper_drawer_main_joint",
        "kitchen_island_right_lower_drawer_main_joint",
        "fridge_area_lower_drawer_main_joint",
        "iai_fridge_door_joint"
    ]

    rospy.logout("START SOME MOVE as Test")
    #  jl1, ja1 = gripper.info_about_joints('sink_area_left_upper_drawer_main_joint')
    #  print(jl1, ja1)
    #open_fridge_door()
    #close_fridge_door()
    #open_kitchen_island_object()
    #close_kitchen_island_object()
    open_oven_door_object()
    #close_oven_door_object()
    #open_sink_area_object()
    #close_sink_area_object()
    #reset_ktichen(list_joint)
    #print test_jacobi_fwd(5, 2, np.deg2rad(45), np.deg2rad(90))
    print("test is done.")

    # set_cart_goal_test()
    rospy.spin()
