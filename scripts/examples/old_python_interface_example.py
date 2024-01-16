import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped

from giskardpy.python_interface.old_python_interface import OldGiskardWrapper

# create goal joint state dictionary
start_joint_state = {'r_elbow_flex_joint': -1.29610152504,
                     'r_forearm_roll_joint': -0.0301682323805,
                     'r_shoulder_lift_joint': 1.20324921318,
                     'r_shoulder_pan_joint': -0.73456435706,
                     'r_upper_arm_roll_joint': -0.70790051778,
                     'r_wrist_flex_joint': -0.10001,
                     'r_wrist_roll_joint': 0.258268529825,

                     'l_elbow_flex_joint': -1.29610152504,
                     'l_forearm_roll_joint': 0.0301682323805,
                     'l_shoulder_lift_joint': 1.20324921318,
                     'l_shoulder_pan_joint': 0.73456435706,
                     'l_upper_arm_roll_joint': 0.70790051778,
                     'l_wrist_flex_joint': -0.1001,
                     'l_wrist_roll_joint': -0.258268529825,

                     'torso_lift_joint': 0.2,
                     'head_pan_joint': 0,
                     'head_tilt_joint': 0}

# init ros node
rospy.init_node('test')

rospy.loginfo('Instantiating Giskard wrapper.')
giskard_wrapper = OldGiskardWrapper()

# Remove everything but the robot.
giskard_wrapper.clear_world()

rospy.loginfo('Combining a joint goal for the arm with a Cartesian goal for the base to reset the pr2.')
# Setting the joint goal
giskard_wrapper.set_joint_goal(start_joint_state)

base_goal = PoseStamped()
base_goal.header.frame_id = 'map'
base_goal.pose.position = Point(0, 0, 0)
base_goal.pose.orientation = Quaternion(0, 0, 0, 1)
# Setting the Cartesian goal.
# Choosing map as root_link will allow Giskard to drive with the pr2.
giskard_wrapper.set_cart_goal(root_link='map', tip_link='base_footprint', goal_pose=base_goal)

# Turn off collision avoidance to make sure that the robot can recover from any state.
giskard_wrapper.allow_all_collisions()
giskard_wrapper.execute()

rospy.loginfo('Setting a Cartesian goal for the right gripper.')
r_goal = PoseStamped()
r_goal.header.frame_id = 'r_gripper_tool_frame'
r_goal.pose.position = Point(-0.2, -0.2, 0.2)
r_goal.pose.orientation = Quaternion(0, 0, 0, 1)
giskard_wrapper.set_cart_goal(root_link='map', tip_link='r_gripper_tool_frame', goal_pose=r_goal)

rospy.loginfo('Setting a Cartesian goal for the left gripper.')
l_goal = PoseStamped()
l_goal.header.frame_id = 'l_gripper_tool_frame'
l_goal.pose.position = Point(0.2, 0.2, 0.2)
l_goal.pose.orientation = Quaternion(0, 0, 0, 1)
giskard_wrapper.set_cart_goal(root_link='map', tip_link='l_gripper_tool_frame', goal_pose=l_goal)

rospy.loginfo('Executing both Cartesian goals at the same time')
giskard_wrapper.execute()

rospy.loginfo('Combining a Cartesian goal with a partial joint goal.')
p = PoseStamped()
p.header.frame_id = 'map'
p.pose.position.x = 0.8
p.pose.position.y = 0.2
p.pose.position.z = 1.0
p.pose.orientation.w = 1

rospy.loginfo('Setting Cartesian goal.')
# Choosing base_footprint as root_link will not include the base, therefore not allowing pr2 to drive.
giskard_wrapper.set_cart_goal(root_link='base_footprint', tip_link='l_gripper_tool_frame', goal_pose=p)

rospy.loginfo('Setting joint goal for only the torso.')
giskard_wrapper.set_joint_goal({'torso_lift_joint': 0.3})

rospy.loginfo('Executing.')
giskard_wrapper.execute()

rospy.loginfo('Setting a pointing goal via the json interface.')
goal_point = PointStamped()
goal_point.header.frame_id = 'r_gripper_tool_frame'

tip = 'high_def_frame'
# The root link torso_lift_link is above the torso joint, therefore the pr2 can't use its torso to achieve the goal.
root = 'torso_lift_link'
pointing_axis = Vector3Stamped()
pointing_axis.header.frame_id = tip
pointing_axis.vector.x = 1

giskard_wrapper.motion_goals.add_motion_goal(motion_goal_class='Pointing',
                                             tip_link=tip,
                                             root_link=root,
                                             goal_point=goal_point,
                                             pointing_axis=pointing_axis)
giskard_wrapper.execute()

rospy.loginfo('Setting a pointing goal via the predefined Giskard wrapper function to look at the left hand.')
goal_point = PointStamped()
goal_point.header.frame_id = 'l_gripper_tool_frame'

tip = 'high_def_frame'
root = 'torso_lift_link'
pointing_axis = Vector3Stamped()
pointing_axis.header.frame_id = tip
pointing_axis.vector.x = 1

giskard_wrapper.set_pointing_goal(tip_link=tip,
                                  root_link=root,
                                  goal_point=goal_point,
                                  pointing_axis=pointing_axis)

rospy.loginfo('Combining it with a goal that makes the right hand point at the left hand.')
tip = 'r_gripper_tool_frame'
root = 'torso_lift_link'
pointing_axis = Vector3Stamped()
pointing_axis.header.frame_id = tip
pointing_axis.vector.x = 1
giskard_wrapper.set_pointing_goal(tip_link='r_gripper_tool_frame',
                                  root_link=root,
                                  goal_point=goal_point,
                                  pointing_axis=pointing_axis)
rospy.loginfo('Execute')
giskard_wrapper.execute()

rospy.loginfo('Spawn a box in the world.')
box_name = 'muh'
box_pose = PoseStamped()
box_pose.header.frame_id = 'r_gripper_tool_frame'
box_pose.pose.orientation.w = 1
giskard_wrapper.add_box(name=box_name,
                        size=(0.2, 0.1, 0.1),
                        pose=box_pose,
                        parent_link='map')
rospy.loginfo('Delete everything but the robot.')
giskard_wrapper.clear_world()

rospy.loginfo('Spawn a box again')
giskard_wrapper.add_box(name=box_name,
                        size=(0.2, 0.1, 0.1),
                        pose=box_pose,
                        parent_link='map')

rospy.loginfo('Attach it to the robot')
giskard_wrapper.update_parent_link_of_group(name=box_name,
                                            parent_link='r_gripper_tool_frame')

rospy.loginfo('Delete only the box.')
giskard_wrapper.remove_group(name=box_name)

rospy.loginfo('Attach a box directly to the robot\'s right gripper.')
giskard_wrapper.add_box(name=box_name,
                        size=(0.2, 0.1, 0.1),
                        pose=box_pose,
                        parent_link='r_gripper_tool_frame')

rospy.loginfo('Set a Cartesian goal for the box')
box_goal = PoseStamped()
box_goal.header.frame_id = box_name
box_goal.pose.position.x = -0.5
box_goal.pose.orientation.w = 1
giskard_wrapper.set_cart_goal(goal_pose=box_goal,
                              tip_link=box_name,
                              root_link='map')
giskard_wrapper.execute()
