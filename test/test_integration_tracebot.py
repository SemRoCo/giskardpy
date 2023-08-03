import giskardpy.utils.tfwrapper as tf
import pytest
from geometry_msgs.msg import PoseStamped

from giskardpy.configs.behavior_tree_config import StandAloneConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.configs.tracy import TracyStandAloneRobotInterface, TracyWorldConfig, TracyCollisionAvoidance
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
        giskard = Giskard(world_config=TracyWorldConfig(),
                          collision_avoidance_config=TracyCollisionAvoidance(),
                          robot_interface_config=TracyStandAloneRobotInterface(),
                          behavior_tree_config=StandAloneConfig(),
                          qp_controller_config=QPControllerConfig())
        super().__init__(giskard)

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
