from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
from itertools import product
from threading import Lock
from typing import List, Union, Dict, Tuple
import numpy as np
import matplotlib.colors as mcolors
import pylab as plt
import rospy
from sortedcontainers import SortedDict
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from giskardpy import identifier
from giskardpy.data_types import JointStates
from giskardpy.god_map import GodMap
from giskardpy.model.joints import Joint, OmniDrive, MovableJoint
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.utils import logging
from giskardpy.utils.utils import cm_to_inch

plot_lock = Lock()


class Trajectory:
    _points: Dict[int, JointStates]

    def __init__(self):
        self.clear()

    def clear(self):
        self._points = OrderedDict()

    def get_exact(self, time):
        return self._points[time]

    def set(self, time: int, point: JointStates):
        if len(self._points) > 0 and list(self._points.keys())[-1] > time:
            raise KeyError('Cannot append a trajectory point that is before the current end time of the trajectory.')
        self._points[time] = point

    def __len__(self) -> int:
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

    def to_msg(self, sample_period: float, start_time: Union[rospy.Duration, float], joints: List[MovableJoint],
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
                free_variables = joint.get_free_variable_names()
                for free_variable in free_variables:
                    if free_variable in traj_point:
                        if i == 0:
                            joint_name = free_variable
                            if isinstance(joint_name, PrefixName):
                                joint_name = joint_name.short_name
                            trajectory_msg.joint_names.append(joint_name)
                        p.positions.append(traj_point[free_variable].position)
                        if fill_velocity_values:
                            p.velocities.append(traj_point[free_variable].velocity)
                    else:
                        raise NotImplementedError('generated traj does not contain all joints')
            trajectory_msg.points.append(p)
        return trajectory_msg

    def to_dict(self, normalize_position: bool = False, filter_0_vel: bool = True) -> Dict[
        Derivatives, Dict[PrefixName, np.ndarray]]:
        data = defaultdict(lambda: defaultdict(list))
        for time, joint_states in self.items():
            for free_variable, joint_state in joint_states.items():
                for derivative, state in enumerate(joint_state.state):
                    data[derivative][free_variable].append(state)
        for derivative, d_data in data.items():
            for free_variable, trajectory in d_data.items():
                d_data[free_variable] = np.array(trajectory)
                if normalize_position and derivative == Derivatives.position:
                    d_data[free_variable] -= (d_data[free_variable].max() + d_data[free_variable].min()) / 2
        if filter_0_vel:
            for free_variable, trajectory in list(data[Derivatives.velocity].items()):
                if abs(trajectory.max() - trajectory.min()) < 1e-5:
                    for derivative, d_data in list(data.items()):
                        del d_data[free_variable]
        for derivative, d_data in data.items():
            data[derivative] = SortedDict(sorted(d_data.items()))
        return data

    @profile
    def plot_trajectory(self,
                        path_to_data_folder: str,
                        sample_period: float,
                        cm_per_second: float = 2.5,
                        normalize_position: bool = False,
                        tick_stride: float = 0.5,
                        file_name: str = 'trajectory.pdf',
                        history: int = 5,
                        height_per_derivative: float = 6,
                        print_last_tick: bool = False,
                        legend: bool = True,
                        hspace: float = 1,
                        y_limits: bool = None,
                        filter_0_vel: bool = True):
        """
        :type tj: Trajectory
        :param controlled_joints: only joints in this list will be added to the plot
        :type controlled_joints: list
        :param velocity_threshold: only joints that exceed this velocity threshold will be added to the plot. Use a negative number if you want to include every joint
        :param cm_per_second: determines how much the x axis is scaled with the length(time) of the trajectory
        :param normalize_position: centers the joint positions around 0 on the y axis
        :param tick_stride: the distance between ticks in the plot. if tick_stride <= 0 pyplot determines the ticks automatically
        """
        cm_per_second = cm_to_inch(cm_per_second)
        height_per_derivative = cm_to_inch(height_per_derivative)
        hspace = cm_to_inch(hspace)
        max_derivative = GodMap().get_data(identifier.max_derivative)
        with plot_lock:
            def ceil(val, base=0.0, stride=1.0):
                base = base % stride
                return np.ceil((float)(val - base) / stride) * stride + base

            def floor(val, base=0.0, stride=1.0):
                base = base % stride
                return np.floor((float)(val - base) / stride) * stride + base

            if len(self._points) <= 0:
                return
            colors = list(mcolors.TABLEAU_COLORS.keys())
            colors.append('k')

            line_styles = ['-', '--', '-.', ':']
            graph_styles = list(product(line_styles, colors))
            color_map: Dict[str, Tuple[str, str]] = defaultdict(lambda: graph_styles[len(color_map) + 1])
            data = self.to_dict(normalize_position, filter_0_vel=filter_0_vel)
            times = np.arange(len(self)) * sample_period

            f, axs = plt.subplots((max_derivative + 1), sharex=True, gridspec_kw={'hspace': hspace})
            f.set_size_inches(w=(times[-1] - times[0]) * cm_per_second, h=(max_derivative + 1) * height_per_derivative)

            plt.xlim(times[0], times[-1])

            if tick_stride > 0:
                first = ceil(times[0], stride=tick_stride)
                last = floor(times[-1], stride=tick_stride)
                ticks = np.arange(first, last, tick_stride)
                ticks = np.insert(ticks, 0, times[0])
                ticks = np.append(ticks, last)
                if print_last_tick:
                    ticks = np.append(ticks, times[-1])
                for derivative in Derivatives.range(start=Derivatives.position, stop=max_derivative):
                    axs[derivative].set_title(str(derivative))
                    axs[derivative].xaxis.set_ticks(ticks)
                    if y_limits is not None:
                        axs[derivative].set_ylim(y_limits)
            else:
                for derivative in Derivatives.range(start=Derivatives.position, stop=max_derivative):
                    axs[derivative].set_title(str(derivative))
                    if y_limits is not None:
                        axs[derivative].set_ylim(y_limits)
            for derivative, d_data in data.items():
                if derivative > max_derivative:
                    continue
                for free_variable, f_data in d_data.items():
                    try:
                        style, color = color_map[str(free_variable)]
                        axs[derivative].plot(times, f_data, color=color,
                                             linestyle=style,
                                             label=free_variable)
                    except KeyError:
                        logging.logwarn(f'Not enough colors to plot all joints, skipping {free_variable}.')
                    except Exception as e:
                        pass
                axs[derivative].grid()

            if legend:
                axs[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')

            axs[-1].set_xlabel('time [s]')

            file_name = path_to_data_folder + file_name
            last_file_name = file_name.replace('.pdf', f'{history}.pdf')

            if os.path.isfile(file_name):
                if os.path.isfile(last_file_name):
                    os.remove(last_file_name)
                for i in np.arange(history, 0, -1):
                    if i == 1:
                        previous_file_name = file_name
                    else:
                        previous_file_name = file_name.replace('.pdf', f'{i - 1}.pdf')
                    current_file_name = file_name.replace('.pdf', f'{i}.pdf')
                    try:
                        os.rename(previous_file_name, current_file_name)
                    except FileNotFoundError:
                        pass
            plt.savefig(file_name, bbox_inches="tight")
            logging.loginfo(f'saved {file_name}')
