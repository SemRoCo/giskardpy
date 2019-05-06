#!/usr/bin/env python
from datetime import datetime
from itertools import product

import rospy
import sys
import pylab as plt
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import numpy as np
from giskardpy import logging


# def plot_trajectory(tj, controlled_joints):
#     """
#     :type tj: Trajectory
#     :param controlled_joints: only joints in this list will be added to the plot
#     :type controlled_joints: list
#     """
#     colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
#     line_styles = [u'', u'--', u'-.']
#     fmts = [u''.join(x) for x in product(line_styles, colors)]
#     positions = []
#     velocities = []
#     times = []
#     names = [x for x in tj._points[0.0].keys() if x in controlled_joints]
#     for time, point in tj.items():
#         positions.append([v.position for j, v in point.items() if j in controlled_joints])
#         velocities.append([v.velocity for j, v in point.items() if j in controlled_joints])
#         times.append(time)
#     positions = np.array(positions)
#     velocities = np.array(velocities).T
#     times = np.array(times)
#
#     f, (ax1, ax2) = plt.subplots(2, sharex=True)
#     ax1.set_title(u'position')
#     ax2.set_title(u'velocity')
#     # positions -= positions.mean(axis=0)
#     for i, position in enumerate(positions.T):
#         ax1.plot(times, position, fmts[i], label=names[i])
#         ax2.plot(times, velocities[i], fmts[i])
#     box = ax1.get_position()
#     ax1.set_ylim(-3, 1)
#     ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
#     box = ax2.get_position()
#     ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])
#
#     # Put a legend to the right of the current axis
#     ax1.legend(loc=u'center', bbox_to_anchor=(1.45, 0))
#
#     plt.savefig(u'trajectory.pdf')

def cb(data):
    """
    :type data: FollowJointTrajectoryActionGoal
    :return:
    """
    colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
    line_styles = [u'', u'--', u'-.']
    fmts = [u''.join(x) for x in product(line_styles, colors)]

    traj = data.goal.trajectory
    positions = []
    velocities = []
    times = []
    names = traj.joint_names
    for jtp in traj.points:  # type: JointTrajectoryPoint
        positions.append(jtp.positions)
        if len(jtp.velocities) > 0:
            velocities.append(jtp.velocities)
        times.append(jtp.time_from_start.to_sec())
    positions = np.array(positions)
    velocities = np.array(velocities)
    times = np.array(times)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.set_title(u'position')
    ax2.set_title(u'velocity')
    # positions -= positions.mean(axis=0)
    for i, position in enumerate(positions.T):
        ax1.plot(times, position, fmts[i], label=names[i])
        if len(velocities) > 0:
            ax2.plot(times, velocities[:, i], fmts[i])
    box = ax1.get_position()
    # ax1.set_ylim(-2, 2)
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc=u'center', bbox_to_anchor=(1.45, 0))
    ax1.grid()
    ax2.grid()
    now = datetime.now()
    plt.savefig(
        u'trajectory_{}-{}-{}-{}-{}-{}.pdf'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    print(u'saved trajectory')
    # plt.show()


if __name__ == '__main__':
    rospy.init_node('vis_joint_traj')
    topic = rospy.get_param('~topic', '/whole_body_controller/follow_joint_trajectory/goal')
    sub = rospy.Subscriber(topic, FollowJointTrajectoryActionGoal, cb, queue_size=10)
    logging.loginfo('running')
    rospy.spin()
