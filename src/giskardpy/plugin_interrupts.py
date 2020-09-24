import matplotlib.pyplot as plt
import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.exceptions import ShakingException
from giskardpy.plugin import GiskardBehavior
# fast
from giskardpy.plugin_goal_reached import make_velocity_threshold


class WiggleCancel(GiskardBehavior):
    def __init__(self, name):
        super(WiggleCancel, self).__init__(name)
        self.amplitude_threshold = self.get_god_map().get_data(identifier.amplitude_threshold)
        self.num_samples_in_fft = self.get_god_map().get_data(identifier.num_samples_in_fft)
        self.frequency_range = self.get_god_map().get_data(identifier.frequency_range)
        self.max_angular_velocity = 10.5
        self.max_linear_velocity = 10.5

    def initialise(self):
        super(WiggleCancel, self).initialise()
        self.js_samples = []
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        self.max_detectable_freq = 1 / (2 * self.sample_period)
        self.min_wiggle_frequency = self.frequency_range * self.max_detectable_freq
        self.keys = []
        self.thresholds = []
        self.velocity_limits = []
        for joint_name, threshold in zip(self.get_robot().controlled_joints,
                                         make_velocity_threshold(self.get_god_map())):
            velocity_limit = self.get_robot().get_joint_velocity_limit_expr_evaluated(joint_name, self.god_map)
            if self.get_robot().is_joint_prismatic(joint_name):
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

    def update(self):
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
        self.detect_shaking(js_samples_array, self.sample_period, self.min_wiggle_frequency,
                            self.amplitude_threshold, self.thresholds, self.velocity_limits, plot)

        return Status.RUNNING

    def make_mask(self, js_samples, moving_thresholds):
        return np.any(js_samples.T > moving_thresholds, axis=0)

    def detect_shaking(self, js_samples, sample_period, min_wiggle_frequency, amplitude_threshold, moving_thresholds,
                       velocity_limits, plot=False):
        N = len(js_samples[0]) - 1
        # remove joints that arent moving
        mask = self.make_mask(js_samples, moving_thresholds)
        velocity_limits = velocity_limits[mask]
        amplitude_thresholds = velocity_limits * amplitude_threshold * N  # acceleration limit is basically vel*2
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
        fft = [np.abs(i.real) for i in fft]

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
                plt.plot(xf, np.abs(yf.real), label=u'real')
                # plt.plot(xf, np.abs(yf.imag), label=u'img')
            plt.show()

        fft = np.array(fft)
        violations = fft[:, freq_idx:].T > amplitude_thresholds
        if np.any(violations):
            filtered_keys = self.keys[mask]
            violation_str = u''
            for i in range(violations.shape[1]):
                if np.any(violations[:, i]):
                    joint = filtered_keys[i]
                    velocity_limit = velocity_limits[i]
                    hertz_str = u', '.join(u'{} hertz: {} > {}'.format(freq[freq_idx:][j],
                                                                     fft[:, freq_idx:].T[:, i][j] / N / velocity_limit,
                                                                     amplitude_threshold) for j, x in
                                         enumerate(violations[:, i]) if x)
                    violation_str += u'\nshaking of joint: \'{}\' at '.format(joint) + hertz_str
            raise ShakingException(u'endless wiggling detected' + violation_str)


# class MaxTrajLength(GiskardBehavior):
#     def update(self):
#         t = self.get_god_map().get_data(identifier.time)
#         sample_period = self.get_god_map().get_data(identifier.sample_period)
#         t = t * sample_period
#         if t > 30:
#             raise InsolvableException(u'trajectory too long')
#
#         return Status.RUNNING
