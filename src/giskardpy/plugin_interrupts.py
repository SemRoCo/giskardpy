from py_trees import Status
from giskardpy.exceptions import PathCollisionException, InsolvableException
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import closest_point_constraint_violated
import numpy as np
from giskardpy import logging

class WiggleCancel(GiskardBehavior):
    def __init__(self, name, final_detection):
        super(WiggleCancel, self).__init__(name)
        self.wiggle_detection_threshold = self.get_god_map().safe_get_data(identifier.wiggle_detection_threshold)
        self.num_samples_in_fft = self.get_god_map().safe_get_data(identifier.num_samples_in_fft)
        self.wiggle_frequency_range = self.get_god_map().safe_get_data(identifier.wiggle_frequency_range)
        self.js_samples = self.get_god_map().safe_get_data(identifier.wiggle_detection_samples)
        self.final_detection = final_detection


    def initialise(self):
        super(WiggleCancel, self).initialise()
        if not self.final_detection:
            self.js_samples.clear()
        self.sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        self.max_detectable_freq = 1 / (2 * self.sample_period)
        self.min_wiggle_frequency = self.wiggle_frequency_range * self.max_detectable_freq


    def update(self):
        if self.final_detection:
            js_samples_array = np.array(self.js_samples.values())
            if(len(js_samples_array) == 0):
                logging.logwarn('sample array was empty during final wiggle detection')
                return Status.SUCCESS

            if len(js_samples_array[0]) < 4:  # if there are less than 4 sample points it makes no sense to try to detect wiggling
                return Status.SUCCESS

            if detect_wiggling(js_samples_array, self.sample_period, self.min_wiggle_frequency, self.wiggle_detection_threshold):
                self.raise_to_blackboard(InsolvableException(u'endless wiggling detected'))

            return Status.SUCCESS
        else:
            latest_points = self.get_god_map().safe_get_data(identifier.joint_states)

            for key in latest_points:
                self.js_samples[key].append(latest_points[key].velocity)

            if len(self.js_samples.values()[0]) < self.num_samples_in_fft:
                return Status.RUNNING

            if len(self.js_samples.values()[0]) > self.num_samples_in_fft:
                for val in self.js_samples.values():
                    del (val[0])

            js_samples_array = np.array(self.js_samples.values())
            if(detect_wiggling(js_samples_array, self.sample_period, self.min_wiggle_frequency, self.wiggle_detection_threshold)):
                raise InsolvableException(u'endless wiggling detected')

            return Status.RUNNING


def detect_wiggling(js_samples, sample_period, min_wiggle_frequency, wiggle_detection_threshold):
    # remove joints that arent moving
    joints_filtered = [i for i in js_samples if
                       np.any(np.abs(i) > 0.1)]

    if len(joints_filtered) == 0:
        return False

    freq = np.fft.rfftfreq(len(js_samples[0]), d=sample_period)

    # find index in frequency list where frequency >= min_wiggle_frequency
    try:
        freq_idx = next(i for i, v in enumerate(freq) if v >= min_wiggle_frequency)
    except StopIteration:
        return False

    fft = np.fft.rfft(joints_filtered, axis=1)
    fft = [np.abs(i.real) for i in fft]

    for j in fft:
        if np.any(j[freq_idx:] > wiggle_detection_threshold):
            return True

    return False



class MaxTrajLength(GiskardBehavior):
    pass


class CollisionCancel(GiskardBehavior):
    def __init__(self, name):
        self.collision_time_threshold = self.get_god_map().safe_get_data(identifier.collision_time_threshold)
        super(CollisionCancel, self).__init__(name)

    def update(self):
        time = self.get_god_map().safe_get_data(identifier.time)
        if time >= self.collision_time_threshold:
            cp = self.get_god_map().safe_get_data(identifier.closest_point)
            if cp is not None and closest_point_constraint_violated(cp, tolerance=1):
                self.raise_to_blackboard(PathCollisionException(
                    u'robot is in collision after {} seconds'.format(self.collision_time_threshold)))
                return Status.SUCCESS
        return Status.FAILURE
