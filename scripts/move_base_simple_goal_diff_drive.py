#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

from giskardpy.python_interface import GiskardWrapper


def call_back(goal: PoseStamped):
    rospy.loginfo('received simple move base goal')
    tip_link = 'base_footprint'
    root_link = 'map'
    giskard.set_json_goal(constraint_type='DiffDriveBaseGoal',
                          tip_link=tip_link, root_link=root_link,
                          goal_pose=goal)
    giskard.allow_all_collisions()
    giskard.plan_and_execute(wait=False)


if __name__ == '__main__':
    try:
        rospy.init_node('move_base_simple_goal')

        giskard = GiskardWrapper()
        sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_back, queue_size=10)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
