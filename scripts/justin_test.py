from copy import deepcopy

from giskardpy.motion_graph.tasks.task import WEIGHT_COLLISION_AVOIDANCE
from giskardpy.python_interface.python_interface import GiskardWrapper
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from tf.transformations import quaternion_from_matrix
from giskardpy.goals.weight_scaling_goals import MaxManipulabilityLinWeight, BaseArmWeightScaling

rospy.init_node('test_justin')
giskard = GiskardWrapper()

giskard.clear_motion_goals_and_monitors()
giskard.world.clear()

right_arm_joints = [
    "right_arm1_joint",
    "right_arm2_joint",
    "right_arm3_joint",
    "right_arm4_joint",
    "right_arm5_joint",
    "right_arm6_joint",
    "right_arm7_joint",
]
better_pose = {
    "torso1_joint": 0.0,
    "torso2_joint": -0.9,
    "torso3_joint": 1.26,
    "torso4_joint": 0.0,
    "head1_joint": 0.0,
    "head2_joint": 0.0,
    "left_arm1_joint": 0.41,
    "left_arm2_joint": -1.64,
    "left_arm3_joint": 0.12,
    "left_arm4_joint": 0.96,
    "left_arm5_joint": 0.71,
    "left_arm6_joint": -0.02,
    "left_arm7_joint": 0.43,
    "right_arm1_joint": 0.6,
    "right_arm2_joint": -1.59,
    "right_arm3_joint": 2.97,
    "right_arm4_joint": -0.99,
    "right_arm5_joint": -2.44,
    "right_arm6_joint": 0.0,
    "right_arm7_joint": 0.0,
}

torso_pose = {
    "torso1_joint": 0.0,
    "torso2_joint": -0.9,
    "torso3_joint": 1.26,
    "torso4_joint": 0.0}

p = PoseStamped()
p.header.frame_id = 'map'
giskard.world.add_urdf(name='kitchen', urdf=rospy.get_param('/kitchen_description'), pose=p, parent_link='map')

pose = PoseStamped()
pose.header.frame_id = 'map'
pose.pose.position = Point(-2.3, -2.5, 0)
pose.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                            [-1, 0, 0, 0],
                                                            [0, 0, 1, 0],
                                                            [0, 0, 0, 1]]))
giskard.motion_goals.add_cartesian_pose(pose, 'base_link', 'map')
giskard.motion_goals.add_joint_position(better_pose)
giskard.add_default_end_motion_conditions()
giskard.execute()

p2 = PoseStamped()
p2.header.frame_id = 'kitchenette'
p2.pose.orientation.w = 1
p2.pose.position = Point(-0.2, 0, 0.27)
giskard.world.add_box('box', (0.1, 0.05, 0.2), p2)

p_grasp = PoseStamped()
p_grasp.header.frame_id = 'box'
p_grasp.pose.orientation = Quaternion(*quaternion_from_matrix([[0, 1, 0, 0],
                                                               [1, 0, 0, 0],
                                                               [0, 0, -1, 0],
                                                               [0, 0, 0, 1]]))
p_grasp.pose.position = Point(0, 0, 0.12)
p_pre = deepcopy(p_grasp)
p_pre.pose.position.z += 0.15

grasp_point = PointStamped()
grasp_point.header.frame_id = p_grasp.header.frame_id
grasp_point.point = p_grasp.pose.position
mon = giskard.monitors.add_cartesian_pose('map', 'r_gripper_tool_frame', p_pre)
giskard.motion_goals.add_cartesian_pose(p_pre, 'r_gripper_tool_frame', 'map', end_condition=mon, name='pre')

giskard.motion_goals.add_cartesian_pose(p_grasp, 'r_gripper_tool_frame', 'map', start_condition=mon, name='grasp')

giskard.motion_goals.add_motion_goal(motion_goal_class=BaseArmWeightScaling.__name__,
                                     root_link='map',
                                     tip_link='r_gripper_tool_frame',
                                     tip_goal=grasp_point,
                                     gain=10000,
                                     arm_joints=right_arm_joints,
                                     base_joints=['brumbrum', 'torso1_joint', 'torso2_joint', 'torso3_joint',
                                                  'torso4_joint'], )
giskard.motion_goals.add_motion_goal(motion_goal_class=MaxManipulabilityLinWeight.__name__,
                                     root_link='torso4',
                                     tip_link='r_gripper_tool_frame')
p_axis = Vector3Stamped()
p_axis.header.frame_id = 'head1'
p_axis.vector.x = 1
giskard.motion_goals.add_pointing(goal_point=grasp_point, tip_link='head1', pointing_axis=p_axis, root_link='map',
                                  weight=WEIGHT_COLLISION_AVOIDANCE)
#giskard.motion_goals.avoid_all_collisions()
#giskard.motion_goals.allow_collision(group2='box')
giskard.add_default_end_motion_conditions()
giskard.execute()
