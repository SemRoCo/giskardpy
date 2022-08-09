from collections import OrderedDict
from typing import List, Optional, Union

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from giskardpy.data_types import JointStates
from giskardpy.model.joints import Joint


class Trajectory:
    def __init__(self):
        self.clear()

    def clear(self):
        self._points = OrderedDict()

    def get_exact(self, time):
        return self._points[time]

    def set(self, time, point: JointStates):
        if len(self._points) > 0 and list(self._points.keys())[-1] > time:
            raise KeyError('Cannot append a trajectory point that is before the current end time of the trajectory.')
        self._points[time] = point

    def __len__(self):
        return len(self._points)

    def get_joint_names(self):
        if len(self) == 0:
            raise IndexError(f'Trajectory is empty and therefore does not contain any joints.')
        return list(self.get_exact(0).keys())

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

    def to_msg(self, sample_period: float, start_time: Union[rospy.Duration, float], joints: List[Joint],
               fill_velocity_values: bool = True) -> JointTrajectory:
        if isinstance(start_time, (int, float)):
            start_time = rospy.Duration(start_time)
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = start_time
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
