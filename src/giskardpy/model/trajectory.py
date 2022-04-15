from collections import OrderedDict
from typing import List

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from giskardpy.model.joints import Joint


class Trajectory(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._points = OrderedDict()

    def get_exact(self, time):
        return self._points[time]

    def set(self, time, point):
        if len(self._points) > 0 and list(self._points.keys())[-1] > time:
            raise KeyError('Cannot append a trajectory point that is before the current end time of the trajectory.')
        self._points[time] = point

    def delete(self, time):
        del self._points[time]

    def delete_last(self):
        self.delete(list(self._points.keys())[-1])

    def get_last(self):
        return list(self._points.values())[-1]

    def items(self):
        return self._points.items()

    def keys(self):
        return self._points.keys()

    def values(self):
        return self._points.values()

    def to_msg(self, sample_period, joints: List[Joint], fill_velocity_values):
        """
        :type traj: giskardpy.data_types.Trajectory
        :return: JointTrajectory
        """
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = rospy.get_rostime() + rospy.Duration(0.5)
        trajectory_msg.joint_names = []
        for i, (time, traj_point) in enumerate(self.items()):
            p = JointTrajectoryPoint()
            p.time_from_start = rospy.Duration(time * sample_period)
            for joint in joints:
                for free_variable in joint.free_variable_list:
                    if free_variable.name in traj_point:
                        if i == 0:
                            trajectory_msg.joint_names.append(str(free_variable.name))
                        p.positions.append(traj_point[free_variable.name].position)
                        if fill_velocity_values:
                            p.velocities.append(traj_point[free_variable.name].velocity)
                    else:
                        raise NotImplementedError('generated traj does not contain all joints')
            trajectory_msg.points.append(p)
        return trajectory_msg