import rospy

from giskardpy.python_interface.python_interface import GiskardWrapper
from my_giskard_config.my_monitors import MyPayloadMonitor

joint_goal1 = {'torso_lift_joint': 0.3}
joint_goal2 = {'torso_lift_joint': 0.1}

rospy.init_node('custom_monitor', anonymous=True)
giskard = GiskardWrapper()

my_monitor = giskard.monitors.add_monitor(monitor_class=MyPayloadMonitor.__name__,
                                          message='muh')
giskard.motion_goals.add_joint_position(goal_state=joint_goal1)
giskard.monitors.add_end_motion(start_condition=my_monitor)
giskard.execute()

