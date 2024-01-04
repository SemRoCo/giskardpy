#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

from giskardpy.data_types import PrefixName
from giskardpy.python_interface.old_python_interface import OldGiskardWrapper


def call_back(goal: PoseStamped):
    rospy.loginfo('received simple move base goal')
    robot_name = giskard.robot_name
    tip_link = str(PrefixName('base_footprint', robot_name))
    root_link = 'map'
    giskard.motion_goals.add_motion_goal(constraint_type='DiffDriveBaseGoal',
                          tip_link=tip_link, root_link=root_link,
                          goal_pose=goal)
    giskard.allow_all_collisions()
    giskard.execute(wait=False)


if __name__ == '__main__':
    try:
        rospy.init_node('move_base_simple_goal')

        giskard = OldGiskardWrapper()
        sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_back, queue_size=10)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
