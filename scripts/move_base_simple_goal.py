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
    goal = transform_pose(root, pose_stamped)
    js = {x_joint: goal.pose.position.x,
          y_joint: goal.pose.position.y,
          z_joint: rotation_from_matrix(quaternion_matrix([goal.pose.orientation.x,
                                                           goal.pose.orientation.y,
                                                           goal.pose.orientation.z,
                                                           goal.pose.orientation.w]))[0]}
    giskard.set_joint_goal(js)
    giskard.plan_and_execute(wait=False)


if __name__ == '__main__':
    try:
        rospy.init_node('move_base_simple_goal')
        init()
        x_joint = 'odom_x_joint'
        y_joint = 'odom_y_joint'
        z_joint = 'odom_z_joint'
        root = 'odom_combined'

        giskard = GiskardWrapper()
        sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, call_back, queue_size=10)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
