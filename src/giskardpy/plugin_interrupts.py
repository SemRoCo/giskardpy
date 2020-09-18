import numpy as np
from py_trees import Status
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import giskardpy.identifier as identifier
from giskardpy.exceptions import InsolvableException
from giskardpy.plugin import GiskardBehavior
from giskardpy import logging


# fast
from giskardpy.plugin_goal_reached import make_velocity_threshold


class WiggleCancel(GiskardBehavior):
    def __init__(self, name, final_detection):
        super(WiggleCancel, self).__init__(name)
        self.amplitude_threshold = self.get_god_map().get_data(identifier.amplitude_threshold)
        self.num_samples_in_fft = self.get_god_map().get_data(identifier.num_samples_in_fft)
        self.frequency_range = self.get_god_map().get_data(identifier.frequency_range)
        self.final_detection = final_detection

    def initialise(self):
        super(WiggleCancel, self).initialise()
        if self.final_detection:
            self.js_samples = self.get_god_map().get_data(identifier.wiggle_detection_samples)
        else:
            self.js_samples = []
            self.get_god_map().set_data(identifier.wiggle_detection_samples, self.js_samples)

        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        self.max_detectable_freq = 1 / (2 * self.sample_period)
        self.min_wiggle_frequency = self.frequency_range * self.max_detectable_freq
        self.keys = []
        self.thresholds = []
        self.velocity_limits = []
        for joint_name, threshold in zip(self.get_robot().controlled_joints, make_velocity_threshold(self.get_god_map())):
            velocity_limit = self.get_robot().get_joint_velocity_limit(joint_name)
            self.keys.append(joint_name)
            self.thresholds.append(threshold)
            self.velocity_limits.append(velocity_limit)
        self.key_set = set(self.keys)
        self.thresholds = np.array(self.thresholds)
        self.velocity_limits = np.array(self.velocity_limits)
        self.js_samples = [[] for _ in range(len(self.keys))]

    def update(self):
        if self.final_detection:
            js_samples_array = np.array(self.js_samples.values())
            if(len(js_samples_array) == 0):
                logging.logwarn('sample array was empty during final wiggle detection')
                return Status.SUCCESS

            if len(js_samples_array[0]) < 4:  # if there are less than 4 sample points it makes no sense to try to detect wiggling
                return Status.SUCCESS

            if detect_shaking(js_samples_array, self.sample_period, self.min_wiggle_frequency, self.amplitude_threshold):
                self.raise_to_blackboard(InsolvableException(u'endless wiggling detected'))

            return Status.SUCCESS
        else:
            latest_points = self.get_god_map().get_data(identifier.joint_states)

            for i, key in enumerate(self.keys):
                self.js_samples[i].append(latest_points[key].velocity)

            if len(self.js_samples[0]) < self.num_samples_in_fft:
                return Status.RUNNING

            if len(self.js_samples[0]) > self.num_samples_in_fft:
                for i in range(len(self.js_samples)):
                    self.js_samples[i].pop(0)

            js_samples_array = np.array(self.js_samples)
            plot=False
            if(detect_shaking(js_samples_array, self.sample_period, self.min_wiggle_frequency,
                              self.amplitude_threshold, self.thresholds, self.velocity_limits, plot)):
                raise InsolvableException(u'endless wiggling detected')

            return Status.RUNNING


def detect_shaking(js_samples, sample_period, min_wiggle_frequency, amplitude_threshold, moving_thresholds,
                   velocity_limits, plot=False):
    N = len(js_samples[0]) -1
    # remove joints that arent moving
    mask = np.any(js_samples.T > moving_thresholds, axis=0)
    amplitude_threshold = velocity_limits[mask] * amplitude_threshold * N # acceleration limit is basically vel*2
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

        x = np.linspace(0, N*sample_period, N)
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

    return np.any(np.array(fft)[:,freq_idx:].T > amplitude_threshold)



class MaxTrajLength(GiskardBehavior):
    def update(self):
        t = self.get_god_map().get_data(identifier.time)
        sample_period = self.get_god_map().get_data(identifier.sample_period)
        t = t * sample_period
        if t > 30:
            raise InsolvableException(u'trajectory too long')

        return Status.RUNNING



# class CollisionCancel(GiskardBehavior):
#     def __init__(self, name):
#         self.collision_time_threshold = self.get_god_map().get_data(identifier.collision_time_threshold)
#         super(CollisionCancel, self).__init__(name)
#
#     def update(self):
#         time = self.get_god_map().get_data(identifier.time)
#         if time >= self.collision_time_threshold:
#             cp = self.get_god_map().get_data(identifier.closest_point)
#             if cp is not None and closest_point_constraint_violated(cp, tolerance=1):
#                 self.raise_to_blackboard(PathCollisionException(
#                     u'robot is in collision after {} seconds'.format(self.collision_time_threshold)))
#                 return Status.SUCCESS
#         return Status.FAILURE
