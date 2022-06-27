#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped

from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils.tfwrapper import init


def call_back(goal: PoseStamped):
    rospy.loginfo('received simple move base goal')
    tip_link = 'base_footprint'
    root_link = 'map'
    pointing_axis = Vector3Stamped()
    pointing_axis.header.frame_id = tip_link
    pointing_axis.vector.x = 1
    goal_point = PointStamped()
    goal_point.header.frame_id = 'map'
    goal_point.point = goal.pose.position
    giskard.set_json_goal(constraint_type='PointingDiffDrive',
                          tip_link=tip_link, root_link=root_link,
                          pointing_axis=pointing_axis,
                          goal_point=goal_point)
    giskard.allow_all_collisions()
    giskard.set_cart_goal(goal_pose=goal, tip_link=tip_link, root_link=root_link,
                          # max_linear_velocity=0.5,
                          # max_angular_velocity=0.5
                          )
    giskard.plan_and_execute(wait=False)


if __name__ == '__main__':
    try:
        rospy.init_node('move_base_simple_goal')
        init()
        # root = 'odom_combined'

        giskard = GiskardWrapper()
        sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_back, queue_size=10)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
