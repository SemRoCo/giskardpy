import giskardpy.utils.tfwrapper as tf
import pytest
from geometry_msgs.msg import PoseStamped, PointStamped

from giskardpy.configs.behavior_tree_config import StandAloneBTConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.qp_controller_config import QPControllerConfig
from giskardpy.configs.iai_robots.tracy import TracyStandAloneRobotInterfaceConfig, TracyWorldConfig, TracyCollisionAvoidanceConfig
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

    better_pose = {
        'left_shoulder_pan_joint': 2.539670467376709,
        'left_shoulder_lift_joint': -1.46823854119096,
        'left_elbow_joint': 2.1197431723224085,
        'left_wrist_1_joint': -1.4825000625899811,
        'left_wrist_2_joint': 5.467689037322998,
        'left_wrist_3_joint': -0.9808381239520472,
        'right_shoulder_pan_joint': 3.7588136196136475,
        'right_shoulder_lift_joint': -1.7489210567870082,
        'right_elbow_joint': -2.054229259490967,
        'right_wrist_1_joint': -1.6140786610045375,
        'right_wrist_2_joint': 0.7295855283737183,
        'right_wrist_3_joint': 3.944669485092163,
    }

    def __init__(self):
        tf.init()
        giskard = Giskard(world_config=TracyWorldConfig(),
                          collision_avoidance_config=TracyCollisionAvoidanceConfig(),
                          robot_interface_config=TracyStandAloneRobotInterfaceConfig(),
                          behavior_tree_config=StandAloneBTConfig(),
                          qp_controller_config=QPControllerConfig())
        super().__init__(giskard)

    def reset(self):
        # self.mujoco_reset()
        self.clear_world()


class TestTracebot:
    def test_place_cylinder(self, better_pose: TracebotTestWrapper):
        cylinder_name = 'C'
        cylinder_height = 0.121
        hole_point = PointStamped()
        hole_point.header.frame_id = 'table'
        hole_point.point.x = 0.7
        hole_point.point.y = -0.25
        pose = PoseStamped()
        pose.header.frame_id = 'r_gripper_tool_frame'
        pose.pose.position.z = cylinder_height / 5
        pose.pose.orientation.w = 1
        better_pose.add_cylinder(name=cylinder_name,
                                 height=cylinder_height,
                                 radius=0.0225,
                                 pose=pose,
                                 parent_link='r_gripper_tool_frame')
        better_pose.dye_group(cylinder_name, (0, 0, 1, 1))

        better_pose.set_json_goal('InsertCylinder',
                                  cylinder_name=cylinder_name,
                                  cylinder_height=0.121,
                                  hole_point=hole_point)
        better_pose.allow_all_collisions()
        better_pose.plan_and_execute()


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
