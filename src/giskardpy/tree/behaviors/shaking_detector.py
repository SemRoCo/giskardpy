import matplotlib.pyplot as plt
import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import ShakingException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time


# fast
# from giskardpy.tree.goal_reached import make_velocity_threshold


class WiggleCancel(GiskardBehavior):
    @profile
    def __init__(self, name):
        super().__init__(name)
        self.amplitude_threshold = self.get_god_map().get_data(identifier.amplitude_threshold)
        self.num_samples_in_fft = self.get_god_map().get_data(identifier.num_samples_in_fft)
        self.frequency_range = self.get_god_map().get_data(identifier.frequency_range)
        self.max_angular_velocity = 10.5
        self.max_linear_velocity = 10.5

    def make_velocity_threshold(self, min_cut_off=0.01, max_cut_off=0.06):
        joint_convergence_threshold = self.god_map.get_data(identifier.joint_convergence_threshold)
        free_variables = self.god_map.get_data(identifier.free_variables)
        thresholds = []
        for free_variable in free_variables:  # type: FreeVariable
            velocity_limit = self.god_map.evaluate_expr(free_variable.get_upper_limit(1))
            velocity_limit *= joint_convergence_threshold
            velocity_limit = min(max(min_cut_off, velocity_limit), max_cut_off)
            thresholds.append(velocity_limit)
        return np.array(thresholds)

    @record_time
    @profile
    def initialise(self):
        super().initialise()
        self.js_samples = []
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        self.max_detectable_freq = 1 / (2 * self.sample_period)
        self.min_wiggle_frequency = self.frequency_range * self.max_detectable_freq
        self.keys = []
        self.thresholds = []
        self.velocity_limits = []
        # FIXME check for free variables and not joints
        for joint_name, threshold in zip(self.world.controlled_joints,
                                         self.make_velocity_threshold()):
            _, velocity_limit = self.world.get_joint_velocity_limits(joint_name)
            if self.world.is_joint_prismatic(joint_name):
                velocity_limit = min(self.max_linear_velocity, velocity_limit)
            else:
                velocity_limit = min(self.max_angular_velocity, velocity_limit)
            self.keys.append(joint_name)
            self.thresholds.append(threshold)
            self.velocity_limits.append(velocity_limit)
        self.keys = np.array(self.keys)
        self.key_set = set(self.keys)
        self.thresholds = np.array(self.thresholds)
        self.velocity_limits = np.array(self.velocity_limits)
        self.js_samples = [[] for _ in range(len(self.keys))]

    @record_time
    @profile
    def update(self):
        prediction_horizon = self.god_map.get_data(identifier.prediction_horizon)
        if prediction_horizon > 1:
            return Status.RUNNING
        latest_points = self.get_god_map().get_data(identifier.joint_states)

        for i, key in enumerate(self.keys):
            self.js_samples[i].append(latest_points[key].velocity)

        if len(self.js_samples[0]) < self.num_samples_in_fft:
            return Status.RUNNING

        if len(self.js_samples[0]) > self.num_samples_in_fft:
            for i in range(len(self.js_samples)):
                self.js_samples[i].pop(0)

        js_samples_array = np.array(self.js_samples)
        plot = False
        try:
            self.detect_shaking(js_samples_array, self.sample_period, self.min_wiggle_frequency,
                                self.amplitude_threshold, self.thresholds, self.velocity_limits, plot)
        except ShakingException as e:
            if self.get_god_map().get_data(identifier.cut_off_shaking):
                trajectory = self.get_god_map().get_data(identifier.trajectory)
                for i in range(self.num_samples_in_fft):
                    trajectory.delete_last()
                # time = self.get_god_map().get_data(identifier.time)
                # self.get_god_map().set_data(identifier.time, len(trajectory.keys()))
                if len(trajectory.keys()) >= self.num_samples_in_fft:
                    logging.loginfo(str(e))
                    logging.loginfo('cutting off last second')
                    return Status.SUCCESS
            raise

        return Status.RUNNING

    def make_mask(self, js_samples, moving_thresholds):
        return np.any(js_samples.T > moving_thresholds, axis=0)

    def detect_shaking(self, js_samples, sample_period, min_wiggle_frequency, amplitude_threshold, moving_thresholds,
                       velocity_limits, plot=False):
        N = len(js_samples[0]) - 1
        # remove joints that arent moving
        mask = self.make_mask(js_samples, moving_thresholds)
        velocity_limits = velocity_limits[mask]
        amplitude_thresholds = velocity_limits * amplitude_threshold  # acceleration limit is basically vel*2
        joints_filtered = js_samples[mask]
        joints_filtered = np.diff(joints_filtered)
        # joints_filtered = (joints_filtered.T - joints_filtered.mean(axis=1)).T

        if len(joints_filtered) == 0:
            return False

        freq = np.fft.rfftfreq(N, d=sample_period)

        # find index in frequency list where frequency >= min_wiggle_frequency
        try:
            freq_idx = next(i for i, v in enumerate(freq) if v >= min_wiggle_frequency)
        except StopIteration:
            return False

        fft = np.fft.rfft(joints_filtered, axis=1)
        fft = [2.0 * np.abs(i)/N for i in fft]
        if plot:
            y = joints_filtered

            x = np.linspace(0, N * sample_period, N)
            y = np.array(y)
            fig, ax = plt.subplots()
            for yy in y:
                ax.plot(x, yy)
            plt.show()

            fig, ax = plt.subplots()
            for i, yy in enumerate(y):
                yf = fft[i]
                # yf = np.fft.rfft(yy)
                xf = np.fft.rfftfreq(N, d=sample_period)
                plt.plot(xf, np.abs(yf.real), label='real')
                # plt.plot(xf, np.abs(yf.imag), label='img')
            plt.show()

        fft = (velocity_limits * np.array(fft).T).T
        violations = fft[:, freq_idx:].T > amplitude_thresholds
        if np.any(violations):
            filtered_keys = self.keys[mask]
            violation_str = ''
            for i in range(violations.shape[1]):
                if np.any(violations[:, i]):
                    joint = filtered_keys[i]
                    velocity_limit = velocity_limits[i]
                    hertz_str = ', '.join('{} hertz: {} > {}'.format(freq[freq_idx:][j],
                                                                     fft[:, freq_idx:].T[:, i][j] / velocity_limit,
                                                                     amplitude_threshold) for j, x in
                                         enumerate(violations[:, i]) if x)
                    violation_str += '\nshaking of joint: \'{}\' at '.format(joint) + hertz_str
            raise ShakingException('endless wiggling detected' + violation_str)