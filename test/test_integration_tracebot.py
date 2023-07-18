import giskardpy.utils.tfwrapper as tf
import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped, PointStamped
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis

from giskardpy.configs.tiago import TiagoMujoco
from giskardpy.configs.tracy import TracyStandAlone
from giskardpy.utils.utils import launch_launchfile
from utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def giskard(request, ros):
    launch_launchfile('package://iai_tracy_description/launch/upload.launch')
    c = TracebotTestWrapper()
    # c = TracebotTestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


class TracebotTestWrapper(GiskardTestWrapper):
    default_pose = {
        'left_shoulder_pan_joint': 0,
        'left_shoulder_lift_joint': -1.57,
        'left_elbow_joint': -1,
        'left_wrist_1_joint': 0,
        'left_wrist_2_joint': 0,
        'left_wrist_3_joint': 0,
        'right_shoulder_pan_joint': 0,
        'right_shoulder_lift_joint': -1.57,
        'right_elbow_joint': 1,
        'right_wrist_1_joint': 0,
        'right_wrist_2_joint': 0,
        'right_wrist_3_joint': 0,
    }

    def __init__(self):
        tf.init()
        # self.mujoco_reset = rospy.ServiceProxy('tracebot/reset', Trigger)
        super().__init__(TracyStandAlone)

    def reset(self):
        # self.mujoco_reset()
        self.clear_world()


class TestCartGoals:
    def test_move_left_hand(self, zero_pose: TracebotTestWrapper):
        tip = 'left_tool0'
        goal = PoseStamped()
        goal.header.frame_id = tip
        goal.pose.position.x = 0.1
        # goal.pose.position.y = 1
        goal.pose.orientation.w = 1
        # goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))

        zero_pose.set_cart_goal(goal, tip_link=tip, root_link='world')
        # zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()
