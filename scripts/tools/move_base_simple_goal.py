#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import rotation_from_matrix, quaternion_matrix

from giskardpy.python_interface import GiskardWrapper
from giskardpy.utils.tfwrapper import transform_pose, init

def call_back(pose_stamped):
    """
    :type pose_stamped: PoseStamped
    """
    rospy.loginfo('received simple move base goal')
    tip_link = 'base_footprint'
    root_link = 'map'
    giskard.add_cartesian_pose_straight(goal_pose=pose_stamped,
                                        tip_link=tip_link,
                                        root_link=root_link)
    giskard.allow_all_collisions()
    giskard.plan_and_execute(wait=False)


if __name__ == '__main__':
    try:
        rospy.init_node('move_base_simple_goal')
        init()

        giskard = GiskardWrapper()
        sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_back, queue_size=10)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
