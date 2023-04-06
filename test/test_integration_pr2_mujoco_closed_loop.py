from copy import deepcopy
from typing import Optional

import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_srvs.srv import Trigger
import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import MoveResult, MoveGoal
from giskardpy.configs.pr2 import PR2_Mujoco, PR2_MujocoRealTime
from test_integration_pr2 import PR2TestWrapper, TestJointGoals, pocky_pose
from utils_for_tests import JointGoalChecker


class PR2TestWrapperMujoco(PR2TestWrapper):
    def __init__(self):
        del self.default_pose['l_gripper_l_finger_joint']
        del self.default_pose['r_gripper_l_finger_joint']
        self.r_tip = 'r_gripper_tool_frame'
        self.l_tip = 'l_gripper_tool_frame'
        self.l_gripper_group = 'l_gripper'
        self.r_gripper_group = 'r_gripper'
        # self.r_gripper = rospy.ServiceProxy('r_gripper_simulator/set_joint_states', SetJointState)
        # self.l_gripper = rospy.ServiceProxy('l_gripper_simulator/set_joint_states', SetJointState)
        self.mujoco_reset = rospy.ServiceProxy('pr2/reset', Trigger)
        self.odom_root = 'odom_combined'
        super().__init__(PR2_MujocoRealTime)

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = tf.get_tf_root()
        p.pose.orientation.w = 1
        self.set_localization(p)
        self.wait_heartbeats()

    def set_localization(self, map_T_odom: PoseStamped):
        pass

    def teleport_base(self, goal_pose, group_name: Optional[str] = None):
        self.allow_all_collisions()
        self.move_base(goal_pose)

    def reset(self):
        self.mujoco_reset()
        super().reset()


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = PR2TestWrapperMujoco()
    request.addfinalizer(c.tear_down)
    return c


class TestJointGoalsMujoco(TestJointGoals):
    def test_joint_goal2(self, zero_pose: PR2TestWrapper):
        js = {
            # 'torso_lift_joint': 0.2999225173357618,
            'head_pan_joint': 0.041880780651479044,
            'head_tilt_joint': -0.37,
            'r_upper_arm_roll_joint': -0.9487714747527726,
            'r_shoulder_pan_joint': -1.0047307505973626,
            'r_shoulder_lift_joint': 0.48736790658811985,
            'r_forearm_roll_joint': -14.895833882874182,
            'r_elbow_flex_joint': -1.392377908925028,
            'r_wrist_flex_joint': -0.4548695149411013,
            'r_wrist_roll_joint': 0.11426798984097819,
            'l_upper_arm_roll_joint': 1.7383062350263658,
            'l_shoulder_pan_joint': 1.8799810286792007,
            'l_shoulder_lift_joint': 0.011627231224188975,
            'l_forearm_roll_joint': 312.67276414458695,
            'l_elbow_flex_joint': -2.0300928925694675,
            'l_wrist_flex_joint': -0.10014623223021513,
            'l_wrist_roll_joint': -6.062015047706399,
        }
        zero_pose.set_joint_goal(js)
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute()


class TestConstraints:

    def test_SetSeedConfiguration(self, zero_pose: PR2TestWrapper):
        zero_pose.set_seed_configuration(seed_configuration=zero_pose.better_pose)
        zero_pose.set_joint_goal(zero_pose.default_pose)
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.CONSTRAINT_INITIALIZATION_ERROR])

class TestActionServerEvents:
    def test_interrupt_way_points1(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(0, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(deepcopy(p), 'base_footprint')
        zero_pose.add_cmd()
        p.pose.position.x += 10
        zero_pose.set_cart_goal(deepcopy(p), 'base_footprint')
        zero_pose.add_cmd()
        p.pose.position.x += 10
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.SUCCESS,
                                                         MoveResult.PREEMPTED,
                                                         MoveResult.PREEMPTED],
                                   stop_after=2)

        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(-0.1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.add_cmd()
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.0, -0.1, -0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.add_cmd()
        p = PoseStamped()
        p.header.frame_id = zero_pose.r_tip
        p.header.stamp = rospy.get_rostime()
        p.pose.position = Point(0.1, 0.1, 0.1)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, zero_pose.r_tip, zero_pose.default_root)

        zero_pose.plan_and_execute()

    def test_interrupt1(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(1, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=1)

    def test_interrupt2(self, zero_pose: PR2TestWrapper):
        p = PoseStamped()
        p.header.frame_id = 'base_footprint'
        p.pose.position = Point(2, 0, 0)
        p.pose.orientation = Quaternion(0, 0, 0, 1)
        zero_pose.set_cart_goal(p, 'base_footprint')
        zero_pose.allow_all_collisions()
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.PREEMPTED], stop_after=6)

    def test_undefined_type(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_all_collisions()
        zero_pose.send_goal(goal_type=MoveGoal.UNDEFINED,
                            expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_empty_goal(self, zero_pose: PR2TestWrapper):
        zero_pose.cmd_seq = []
        zero_pose.plan_and_execute(expected_error_codes=[MoveResult.INVALID_GOAL])

    def test_plan_only(self, zero_pose: PR2TestWrapper):
        zero_pose.allow_self_collision()
        zero_pose.set_joint_goal(pocky_pose, check=False)
        zero_pose.add_goal_check(JointGoalChecker(zero_pose, zero_pose.default_pose))
        zero_pose.send_goal(goal_type=MoveGoal.PLAN_ONLY)