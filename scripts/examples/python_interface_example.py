import rospy
from geometry_msgs.msg import PoseStamped

from giskardpy.goals.joint_goals import JointPositionList
from giskardpy.monitors.joint_monitors import JointGoalReached
from giskardpy.python_interface.python_interface import GiskardWrapper

# %% Define goals for later
right_arm_goal = {'r_shoulder_pan_joint': -1.7125,
                  'r_shoulder_lift_joint': -0.25672,
                  'r_upper_arm_roll_joint': -1.46335,
                  'r_elbow_flex_joint': -2.12,
                  'r_forearm_roll_joint': 1.76632,
                  'r_wrist_flex_joint': -0.10001,
                  'r_wrist_roll_joint': 0.05106}

left_arm_goal = {'l_shoulder_pan_joint': 1.9652,
                 'l_shoulder_lift_joint': - 0.26499,
                 'l_upper_arm_roll_joint': 1.3837,
                 'l_elbow_flex_joint': -2.12,
                 'l_forearm_roll_joint': 16.99,
                 'l_wrist_flex_joint': - 0.10001,
                 'l_wrist_roll_joint': 0}

base_goal = PoseStamped()
base_goal.header.frame_id = 'map'
base_goal.pose.position.x = 2
base_goal.pose.orientation.w = 1

# %% init ros node and Giskard Wrapper.
# This assumes that roslaunch giskardpy giskardpy_pr2_standalone.launch is running.
rospy.init_node('test')

rospy.loginfo('Instantiating Giskard wrapper.')
giskard_wrapper = GiskardWrapper()

# %% Remove everything but the robot.
# All world related operations are grouped under giskard_wrapper.world.
giskard_wrapper.world.clear()

# %% Monitors observe something and turn to True, if the condition is met. They don't cause any motions.
# All monitor related operations are grouped under giskard_wrapper.monitors.
# Let's define a few
# This one turns True when the length of the current trajectory % mod = 0
alternator = giskard_wrapper.monitors.add_alternator(mod=2)
# This one sleeps and then turns True
sleep1 = giskard_wrapper.monitors.add_sleep(1, name='sleep1')
# This prints a message and then turns True.
# With start_condition you can define which monitors need to be True in order for this one to become active
print1 = giskard_wrapper.monitors.add_print(message=f'{sleep1} done', start_condition=sleep1)
# You can also write logical expressions using "and", "or" and "not" to combine multiple monitors
sleep2 = giskard_wrapper.monitors.add_sleep(1.5, name='sleep2', start_condition=f'{print1} or not {sleep1}')

# %% Now Let's define some motion goals.
# We want to reach two joint goals, so we first define monitors for checking that end condition.
right_monitor = giskard_wrapper.monitors.add_joint_position(goal_state=right_arm_goal,
                                                            name='right pose reached',
                                                            start_condition=sleep1)
# You can use add_motion_goal to add any monitor implemented in giskardpy.monitor.
# All remaining parameters are forwarded to the __init__ function of that class.
# All specialized add_ functions are just wrappers for add_monitor.
left_monitor = giskard_wrapper.monitors.add_monitor(monitor_class=JointGoalReached.__name__,
                                                    goal_state=left_arm_goal,
                                                    name='left pose reached',
                                                    start_condition=sleep1,
                                                    threshold=0.01)

# We set two separate motion goals for the joints of the left and right arm.
# All motion goal related operations are groups under giskard_wrapper.motion_goals.
# The one for the right arm starts when the sleep2 monitor is done and ends, when the right_monitor is done,
# meaning it continues until the joint goal was reached.
giskard_wrapper.motion_goals.add_joint_position(goal_state=right_arm_goal,
                                                name='right pose',
                                                start_condition=sleep2,
                                                end_condition=right_monitor)
# You can use add_motion_goal to add any motion goal implemented in giskardpy.goals.
# All remaining parameters are forwarded to the __init__ function of that class.
giskard_wrapper.motion_goals.add_motion_goal(motion_goal_class=JointPositionList.__name__,
                                             goal_state=left_arm_goal,
                                             name='left pose',
                                             end_condition=left_monitor)

# %% Now let's define a goal for the base, 2m in front of it.
# First we define a monitor which checks if that pose was reached.
base_monitor = giskard_wrapper.monitors.add_cartesian_pose(root_link='map',
                                                           tip_link='base_footprint',
                                                           goal_pose=base_goal)

# and then we define a motion goal for it.
# The hold_condition causes the motion goal to hold as long as the condition is True.
# In this case, the cart pose is halted if time % 2 == 1 and active if time % 2 == 0.
giskard_wrapper.motion_goals.add_cartesian_pose(root_link='map',
                                                tip_link='base_footprint',
                                                goal_pose=base_goal,
                                                hold_condition=f'not {alternator}',
                                                end_condition=base_monitor)

# %% Define when the motion should end.
# Usually you'd use the local minimum reached monitor for this.
# Most monitors also have a stay_true parameter (when it makes sense), with reasonable default values.
# In this case, we don't want the local minimum reached monitor to stay True, because it might get triggered during
# the sleeps and therefore set it to False.
local_min = giskard_wrapper.monitors.add_local_minimum_reached(stay_true=False)

# Giskard will only end the motion generation and return Success, if an end monitor becomes True.
# We do this by defining one that gets triggered, when a local minimum was reached, sleep2 is done and the motion goals
# were reached.
giskard_wrapper.monitors.add_end_motion(start_condition=' and '.join([local_min,
                                                                      sleep2,
                                                                      right_monitor,
                                                                      left_monitor,
                                                                      base_monitor]))
# It's good to also add a cancel condition in case something went wrong and the end motion monitor is unable to become
# True. Currently, the only predefined specialized cancel monitor is max trajectory length.
# Alternative you can use monitor.add_cancel_motion similar to end_motion.
giskard_wrapper.monitors.add_max_trajectory_length(120)
# Lastly we allow all collisions
giskard_wrapper.motion_goals.allow_all_collisions()
# And execute the goal.
rospy.loginfo('Sending first goal.')
giskard_wrapper.execute()
rospy.loginfo('First goal finished.')

# %% manipulate world
box_name = 'muh'
box_pose = PoseStamped()
box_pose.header.frame_id = 'r_gripper_tool_frame'
box_pose.pose.orientation.w = 1
rospy.loginfo('Add box.')
giskard_wrapper.world.add_box(name=box_name,
                              size=(0.2, 0.1, 0.1),
                              pose=box_pose,
                              parent_link='map')
rospy.loginfo('Clear world.')
giskard_wrapper.world.clear()

rospy.loginfo('Add box again.')
giskard_wrapper.world.add_box(name=box_name,
                              size=(0.2, 0.1, 0.1),
                              pose=box_pose,
                              parent_link='map')

rospy.loginfo('Attach box at gripper.')
giskard_wrapper.world.update_parent_link_of_group(name=box_name,
                                                  parent_link='r_gripper_tool_frame')

rospy.loginfo('Delete box.')
giskard_wrapper.world.remove_group(name=box_name)

rospy.loginfo('Add a new box directly at gripper.')
giskard_wrapper.world.add_box(name=box_name,
                              size=(0.2, 0.1, 0.1),
                              pose=box_pose,
                              parent_link='r_gripper_tool_frame')

# All objects added to the world can be used as root or tip links in most motion goals or monitors.
# In this case we use the box name to set a goal for the box attached to the robot.
box_goal = PoseStamped()
box_goal.header.frame_id = box_name
box_goal.pose.position.x = 0.5
box_goal.pose.orientation.w = 1
giskard_wrapper.motion_goals.add_cartesian_pose(goal_pose=box_goal,
                                                tip_link=box_name,
                                                root_link='map')

# If you don't want to create complicated monitor/motion goal chains, the default ending conditions might be sufficient.
giskard_wrapper.add_default_end_motion_conditions()
rospy.loginfo('Send cartesian goal for box.')
giskard_wrapper.execute()
rospy.loginfo('Done.')
