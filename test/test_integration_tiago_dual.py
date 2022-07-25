import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped, PointStamped, Point
from std_srvs.srv import Trigger
from tf.transformations import quaternion_about_axis

from giskardpy.configs.tiago import TiagoMujoco
from giskardpy.goals.goal import WEIGHT_ABOVE_CA
from giskardpy.utils.utils import resolve_ros_iris
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
        self.mujoco_reset = rospy.ServiceProxy('reset', Trigger)
        super().__init__(TiagoMujoco)

    def move_base(self, goal_pose):
        tip_link = 'base_footprint'
        root_link = 'map'
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = tip_link
        pointing_axis.vector.x = 1
        goal_point = PointStamped()
        goal_point.header.frame_id = 'map'
        goal_point.point = goal_pose.pose.position
        self.set_json_goal(constraint_type='PointingDiffDrive',
                                tip_link=tip_link, root_link=root_link,
                                pointing_axis=pointing_axis,
                                goal_point=goal_point)
        self.set_cart_goal(goal_pose=goal_pose, tip_link=tip_link, root_link=root_link)
        self.allow_all_collisions()
        self.plan_and_execute()

    def reset(self):
        self.mujoco_reset()
        self.clear_world()
        self.reset_base()


class TestCartGoals:
    def test_drive(self, zero_pose: TiagoTestWrapper):

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 1
        goal.pose.position.y = 1
        # goal.pose.orientation.w = 1
        goal.pose.orientation = Quaternion(*quaternion_about_axis(np.pi / 4, [0, 0, 1]))

        zero_pose.move_base(goal)
        # zero_pose.set_translation_goal(goal, 'base_footprint', 'odom')
        # zero_pose.plan_and_execute()

    def test_drive2(self, zero_pose: TiagoTestWrapper):

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(0.489, -0.598, 0.000)
        goal.pose.orientation.w = 1
        zero_pose.move_base(goal)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position = Point(-0.026,  0.569, 0.000)
        goal.pose.orientation = Quaternion(0,0,0.916530200374776,0.3999654882623912)
        zero_pose.move_base(goal)


class TestCollisionAvoidance:
    def test_left_arm(self, zero_pose: TiagoTestWrapper):
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'arm_left_3_link'
        box_pose.pose.position.z = 0.07
        box_pose.pose.position.x = 0.1
        box_pose.pose.orientation.w = 1
        # zero_pose.add_box('box',
        #                   size=(0.05,0.05,0.05),
        #                   pose=box_pose)
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'arm_left_5_link'
        box_pose.pose.position.z = 0.07
        box_pose.pose.position.y = -0.1
        box_pose.pose.orientation.w = 1
        # zero_pose.add_box('box2',
        #                   size=(0.05,0.05,0.05),
        #                   pose=box_pose)
        # zero_pose.allow_self_collision()
        zero_pose.plan_and_execute()


    def test_load_negative_scale(self, zero_pose: TiagoTestWrapper):
        mesh_path = 'package://tiago_description/meshes/arm/arm_3_collision.dae'
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.3
        box_pose.pose.orientation.w = 1
        zero_pose.add_mesh('meshy',
                           mesh=mesh_path,
                           pose=box_pose,
                           scale=(1, 1, -1),
                           )
        box_pose = PoseStamped()
        box_pose.header.frame_id = 'base_link'
        box_pose.pose.position.x = 0.6
        box_pose.pose.orientation.w = 1
        zero_pose.add_mesh('meshy2',
                           mesh=mesh_path,
                           pose=box_pose,
                           scale=(1, 1, 1),
                           )
        zero_pose.plan_and_execute()