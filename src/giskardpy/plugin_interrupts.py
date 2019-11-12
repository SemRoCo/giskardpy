from py_trees import Status
from time import time

from giskardpy import logging
from giskardpy.exceptions import PathCollisionException, InsolvableException
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import closest_point_constraint_violated
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#fast

class WiggleCancel(GiskardBehavior):
    def __init__(self, name):
        super(WiggleCancel, self).__init__(name)
        self.fft_duration = self.get_god_map().safe_get_data(identifier.fft_duration)
        self.wiggle_detection_threshold = self.get_god_map().safe_get_data(identifier.wiggle_detection_threshold)
        self.min_wiggle_frequency = self.get_god_map().safe_get_data(identifier.min_wiggle_frequency)
        self.sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        self.num_points_in_fft = int(self.fft_duration / self.sample_period)


    def initialise(self):
        self.past_joint_states = set()
        self.joint_dict = defaultdict(list)
        super(WiggleCancel, self).initialise()

    def update(self):
        latest_points = self.get_god_map().safe_get_data(identifier.joint_states)

        for key in latest_points:
            self.joint_dict[key].append(latest_points[key].velocity)

        if len(self.joint_dict.values()[0]) < self.num_points_in_fft:
            return Status.RUNNING

        if len(self.joint_dict.values()[0]) > self.num_points_in_fft:
            for val in self.joint_dict.values():
                del (val[0])

        # remove joints that arent moving
        joints_filtered = np.array(self.joint_dict.values())
        joints_filtered = [i for i in joints_filtered if
                           np.any(np.abs(i) > 0.1)]

        if len(joints_filtered) == 0:
            return Status.RUNNING

        freq = np.fft.rfftfreq(self.num_points_in_fft, d=self.sample_period)

        # find index in frequency list where frequency >= min_wiggle_frequency
        freq_idx = next(i for i,v in enumerate(freq) if v >= self.min_wiggle_frequency)

        fft = np.fft.rfft(joints_filtered, axis=1)
        fft = [np.abs(i.real) for i in fft]

        for j in fft:
            if np.any(j[freq_idx:] > self.wiggle_detection_threshold):
                raise InsolvableException(u'endless wiggling detected')

        return Status.RUNNING



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
