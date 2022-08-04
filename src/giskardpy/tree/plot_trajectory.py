from copy import deepcopy

import numpy as np
import rospy
import yaml
import matplotlib.pyplot as plt
from py_trees import Status

from giskardpy import identifier
from giskardpy.utils.logging import logwarn
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.utils.utils import plot_trajectory, convert_dictionary_to_ros_message


class PlotTrajectory(GiskardBehavior):
    def __init__(self, name, enabled, path_enabled, history, velocity_threshold, scaling, normalize_position, tick_stride, order=4):
        super(PlotTrajectory, self).__init__(name)
        self.order = order
        self.path_enabled = path_enabled
        self.history = history
        self.velocity_threshold = velocity_threshold
        self.scaling = scaling
        self.normalize_position = normalize_position
        self.tick_stride = tick_stride
        self.path_to_data_folder = self.get_god_map().get_data(identifier.data_folder)

    def update(self):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        if trajectory:
            if self.path_enabled:
                cmd = self.god_map.get_data(identifier.next_move_goal)
                for c in cmd.constraints:
                    c_d = yaml.load(c.parameter_value_pair)
                    if 'goals' in c_d and 'tip_link' in c_d:
                        path = c_d['goals']
                        if 'start' in c_d and c_d['start'] is not None:
                            path.insert(0, c_d['start'])
                        self.plot_path(path, c_d['tip_link'])
                        return Status.SUCCESS
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            # controlled_joints = self.god_map.get_data(identifier.controlled_joints)
            controlled_joints = list(trajectory.get_exact(0).keys())
            try:
                plot_trajectory(trajectory, controlled_joints, self.path_to_data_folder, sample_period, self.order,
                                self.velocity_threshold, self.scaling, self.normalize_position, self.tick_stride,
                                history=self.history)
            except Exception as e:
                logwarn(u'failed to save trajectory pdf')
        return Status.SUCCESS

    def plot_path(self, path, link, round_until_decimal = 4):
        trajectory = self.get_god_map().get_data(identifier.trajectory)
        path_positions = [convert_dictionary_to_ros_message(p).pose.position for p in path]

        def calc_closest_dists(points, fks) -> [float]:
            dists = list()
            c_fks = deepcopy(fks)
            for point in points:
                closest_t = c_fks[0]
                for t in c_fks:
                    if np.linalg.norm(point - t) < np.linalg.norm(point - closest_t):
                        closest_t = t
                dists.append(np.linalg.norm(point - closest_t))
            return dists

        def calc_max_dist(path_arrs, trajectory_arrs, round_decimal=4) -> float:
            return round(max(calc_closest_dists(path_arrs, trajectory_arrs)), round_decimal)

        def calc_sum_dist(path_arrs, trajectory_arrs, round_decimal=4) -> float:
            return round(sum(calc_closest_dists(path_arrs, trajectory_arrs)), round_decimal)

        # Accumulate computed trajectory poses of the given tip link
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
        p_arrs = list([np.array(e) for e in zip(p_x, p_y, p_z)])
        t_arrs = list([np.array(e) for e in zip(x, y, z)])
        max_dist = calc_max_dist(p_arrs, t_arrs, round_decimal=round_until_decimal)
        summed_dist = calc_sum_dist(p_arrs, t_arrs, round_decimal=round_until_decimal)
        # prepare some coordinates
        # draw kitchen island as box
        ax.set(xlabel='y (m)', ylabel='x (m)', zlabel='z (m)',
               title=f'Planned Path and Executed Trajectory\n '
                     f'with Maximal and Summed Deviation\n'
                     f'of {max_dist}m and {summed_dist}m')
        plt.savefig('/home/thomas/master_thesis/benchmarking_data/'
                    'path_following/box/navi_5/{}_path_following.png'.format(rospy.get_time()))
        self.log_path_trajectory(max_dist, summed_dist)
        plt.show()

    def log_path_trajectory(self, max_dev, summed_dev, filepath='/home/thomas/2d_axis.txt'):
        try:
            path_cost = self.god_map.get_data(identifier.rosparam + ['path_cost'])
        except KeyError:
            path_cost = None
        try:
            path_time = self.god_map.get_data(identifier.rosparam + ['path_time'])
        except KeyError:
            path_time = None
        planning_time = self.get_god_map().get_data(identifier.time)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        length = planning_time * sample_period
        solve_time = self.get_runtime()
        vs = [path_time, path_cost, length, solve_time, max_dev, summed_dev]
        with open(filepath) as fp:
            lines = fp.read().splitlines()
        if len(lines) == 0:
            lines = ['PathTime:', 'PathLength:', 'TrajectoryLength:', 'TrajectorySolvingTime:', 'PathMaxDev:', 'PathSummedDev:']
        with open(filepath, "w") as fp:
            for line, v in zip(lines, vs):
                print(str(line) + str(v) + ',', file=fp)

    def filter_fp_out(self, from_fp, to_fp, false_dev=0.5):
        with open("/home/thomas/test.txt") as fp:
            lines = fp.read().splitlines()
        inds = list()
        for line in lines:
            if 'PathSummedDev' in line:
                items = line.split(',')
                for i in range(len(items)):
                    try:
                        f_i = float(items[i])
                    except Exception:
                        continue
                    if f_i > 0.5:
                        inds.append(i)
        with open("/home/thomas/test_rmv_err.txt", "w") as fp:
            for line in lines:
                items = line.split(',')
                for i in range(len(items)):
                    if i in inds:
                        continue
                    else:
                        print(str(items[i]) + ",", file=fp)

