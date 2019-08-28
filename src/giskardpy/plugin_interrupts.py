from py_trees import Status
from time import time

from giskardpy import logging
from giskardpy.exceptions import PathCollisionException, InsolvableException
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import closest_point_constraint_violated
import numpy as np
import matplotlib.pyplot as plt
import math

class WiggleCancel(GiskardBehavior):
    def __init__(self, name):
        super(WiggleCancel, self).__init__(name)
        self.wiggle_precision1 = self.get_god_map().safe_get_data(identifier.wiggle_precision_threshold)

    def initialise(self):
        self.past_joint_states = set()
        self.joint_dict = {}
        super(WiggleCancel, self).initialise()

    def update(self):
        current_js = self.get_god_map().safe_get_data(identifier.joint_states)
        sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        current_time = self.get_god_map().safe_get_data(identifier.time) * sample_period
        rounded_js = self.round_js(current_js)
        # TODO make 1 a parameter
        # if current_time >= 18:
        self.check_fft()
        if current_time >= 1 and rounded_js in self.past_joint_states:
            current_max_joint_vel = np.abs([v.velocity for v in current_js.values()]).max()
            # TODO this threshold should depend on the joint type
            if current_max_joint_vel < 0.25:
                logging.loginfo(u'stopped by wiggle detector')
                logging.loginfo(u'found goal trajectory with length {}s in {}s'.format(current_time,
                                                                                       time() - self.get_blackboard().runtime))
                return Status.SUCCESS
            logging.loginfo(u'current max joint vel = {}'.format(current_max_joint_vel))
            raise InsolvableException(u'endless wiggling detected')
        self.past_joint_states.add(rounded_js)
        return Status.RUNNING

    def round_js(self, js):
        """
        :param js: joint_name -> SingleJointState
        :type js: dict
        :return: a sequence of all the rounded joint positions
        :rtype: tuple
        """
        return tuple(round(x.position, self.wiggle_precision1) for x in js.values())

    #@profile
    def check_fft(self):
        fft_duration = 3
        wiggle_sensitivity = 10
        min_wiggle_frequency = 3

        trajectory = self.get_god_map().safe_get_data(identifier.trajectory)

        sample_period = self.get_god_map().safe_get_data(identifier.sample_period)
        num_points = int(fft_duration/sample_period)

        latest_points = trajectory._points[next(reversed(trajectory._points))]

        if len(self.joint_dict.keys()) == 0:
            for key in latest_points.keys():
                self.joint_dict[key] = []

        for key in latest_points.keys():
            self.joint_dict[key].append(latest_points[key].velocity)

        if len(self.joint_dict.values()[0]) < num_points:
            return Status.RUNNING

        if len(self.joint_dict.values()[0]) > num_points:
            for val in self.joint_dict.values():
                del(val[0])

        # remove joints that arent moving
        joints_filtered = [self.joint_dict[key] for key in self.joint_dict.keys() if np.any(abs(np.array(self.joint_dict[key])) > 0.1)]

        if len(joints_filtered) == 0:
            return Status.RUNNING

        freq = np.fft.fftfreq(num_points, d=sample_period)
        freq = freq[:(len(freq) / 2)]  # remove everything < 0

        # find index in frequency list where frequency >= min_wiggle_frequency
        freq_idx = 0
        for i in range(len(freq)):
            if freq[i] >= min_wiggle_frequency:
                freq_idx = i
                break

        fft = np.fft.fft(joints_filtered, axis=1)
        fft = [abs(i.real[:len(i)/2]) for i in fft] # remove everything < 0

        #o = 0
        for l in fft:
            if np.any(l[freq_idx:] > wiggle_sensitivity):
                #plt.clf()
                #plt.plot(joints_filtered[o])
                #plt.savefig("1.svg")
                #plt.clf()
                #plt.plot(freq, l)
                #plt.savefig("2.svg")
                raise InsolvableException(u'endless wiggling detected')
            #o = o+1
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
