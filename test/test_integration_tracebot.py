import giskardpy.utils.tfwrapper as tf
import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped, PointStamped
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis

from giskardpy.configs.tiago import TiagoMujoco
from giskardpy.configs.tracebot import TracebotMujoco
from utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = TracebotTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TracebotTestWrapper(GiskardTestWrapper):
    default_pose = {
        'tracebot_left_arm_shoulder_pan_joint': 0,
        'tracebot_left_arm_shoulder_lift_joint': -1.57,
        'tracebot_left_arm_elbow_joint': 1,
        'tracebot_left_arm_wrist_1_joint': 0,
        'tracebot_left_arm_wrist_2_joint': 0,
        'tracebot_left_arm_wrist_3_joint': 0,
        'tracebot_right_arm_shoulder_pan_joint': 0,
        'tracebot_right_arm_shoulder_lift_joint': -1.57,
        'tracebot_right_arm_elbow_joint': -1,
        'tracebot_right_arm_wrist_1_joint': 0,
        'tracebot_right_arm_wrist_2_joint': 0,
        'tracebot_right_arm_wrist_3_joint': 0,
    }

    def __init__(self):
        tf.init()
        self.mujoco_reset = rospy.ServiceProxy('tracebot/reset', Trigger)
        super().__init__(TracebotMujoco)

    def reset(self):
        self.mujoco_reset()
        self.clear_world()


class TestCartGoals(object):
    def test_drive(self, zero_pose: TracebotTestWrapper):
        tip = 'tracebot_left_gripper_tool_frame'
        goal = PoseStamped()
        goal.header.frame_id = tip
        goal.pose.position.x = 0.05
        # goal.pose.position.y = 1
        goal.pose.orientation.w = 1
        # goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))

        zero_pose.set_cart_goal(goal, tip_link=tip, root_link='tracebot_base_link')
        zero_pose.plan_and_execute()
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()
