import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from std_srvs.srv import Trigger, TriggerRequest
from tf.transformations import quaternion_about_axis

from giskardpy.configs.tiago import Tiago
from utils_for_tests import GiskardTestWrapper


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = TiagoTestWrapper()
    request.addfinalizer(c.tear_down)
    return c


class TiagoTestWrapper(GiskardTestWrapper):
    default_pose = {
        'torso_lift_joint': 2.220446049250313e-16,
        'head_1_joint': 0.0,
        'head_2_joint': 0.0,
        'arm_left_1_joint': 0.0,
        'arm_left_2_joint': 0.0,
        'arm_left_3_joint': 0.0,
        'arm_left_4_joint': 0.0,
        'arm_left_5_joint': 0.0,
        'arm_left_6_joint': 0.0,
        'arm_left_7_joint': 0.0,
        'arm_right_1_joint': 0.0,
        'arm_right_2_joint': 0.0,
        'arm_right_3_joint': 0.0,
        'arm_right_4_joint': 0.0,
        'arm_right_5_joint': 0.0,
        'arm_right_6_joint': 0.0,
        'arm_right_7_joint': 0.0,
    }

    def __init__(self):
        self.mujoco_reset = rospy.ServiceProxy('tiago/reset', Trigger)
        super().__init__(Tiago)

    def move_base(self, goal_pose):
        self.allow_all_collisions()
        self.set_cart_goal(goal_pose=goal_pose, tip_link='base_footprint', root_link='map')
        self.plan_and_execute()

    def reset(self):
        self.mujoco_reset()
        self.clear_world()
        self.reset_base()


class TestCartGoals(object):
    def test_drive(self, zero_pose: TiagoTestWrapper):
        zero_pose.allow_all_collisions()
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1
        # goal.pose.position.y = 1
        goal.pose.orientation.w = 1
        # goal.pose.orientation = Quaternion(*quaternion_about_axis(1, [0, 0, 1]))
        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()
