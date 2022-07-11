from copy import deepcopy
from threading import Thread

import numpy as np
import yaml
from matplotlib import pyplot as plt
from py_trees import Status

from giskardpy import identifier
from giskardpy.utils.logging import logwarn
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.utils import plot_trajectory, convert_dictionary_to_ros_message


class PlotTrajectory(GiskardBehavior):
    plot_thread: Thread

    def __init__(self, name, enabled, path_enabled, wait=False, **kwargs):
        super(PlotTrajectory, self).__init__(name)
        self.wait = wait
        self.path_enabled = path_enabled
        self.kwargs = kwargs
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    @profile
    def initialise(self):
        self.plot_thread = Thread(target=self.plot)
        self.plot_thread.start()

    def plot(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        if trajectory:
            if self.path_enabled:
                cmd = self.god_map.get_data(identifier.next_move_goal)
                for c in cmd.constraints:
                    c_d = yaml.load(c.parameter_value_pair)
                    if 'goals' in c_d and 'tip_link' in c_d:
                        self.plot_path(c_d['goals'], c_d['tip_link'])
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(tj=trajectory,
                                controlled_joints=controlled_joints,
                                path_to_data_folder=self.path_to_data_folder,
                                sample_period=sample_period,
                                diff_after=2,
                                **self.kwargs)
            except Exception as e:
                logwarn(e)
                logwarn('failed to save trajectory.pdf')

    def plot_path(self, path, link):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        path_positions = [convert_dictionary_to_ros_message(p).pose.position for p in path]

        def calc_max_dist(path, trajectory):
            dists = list()
            for point in path:
                closest_t = trajectory[0]
                for t in trajectory:
                    if np.linalg.norm(point - t) < np.linalg.norm(point - closest_t):
                        closest_t = t
                dists.append(np.linalg.norm(point - closest_t))
            return max(dists)

        x = list()
        y = list()
        z = list()
        old_js = deepcopy(self.collision_scene.world.state)
        for p in trajectory.values():
            self.collision_scene.world.state = p
            self.collision_scene.world.notify_state_change()
            fk = self.collision_scene.world.get_fk('map', link)[:3, 3]
            x.append(fk[0])
            y.append(fk[1])
            z.append(fk[2])
        self.collision_scene.world.state = old_js

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10, 8))
        ax.plot(x, y, z, color='red', linestyle='-', linewidth=5)
        p_x = [pp.x for pp in path_positions]
        p_y = [pp.y for pp in path_positions]
        p_z = [pp.z for pp in path_positions]
        ax.plot(p_x, p_y, p_z, color='c', linestyle='dotted', linewidth=5)
        ax.legend(['Executed Trajectory', 'Planned Path'])
        d = calc_max_dist(list([np.array(e) for e in zip(p_x, p_y, p_z)]),
                          list([np.array(e) for e in zip(x, y, z)]))
        ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
               title='Planned Path and Executed Trajectory\n '
                     'with Maximal Deviation of {}m'.format(round(d, 4)))

        plt.show()

    @profile
    def update(self):
        if self.wait and self.plot_thread.is_alive():
                return Status.RUNNING
        return Status.SUCCESS
