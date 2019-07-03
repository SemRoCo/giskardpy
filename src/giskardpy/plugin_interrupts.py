from py_trees import Status
from time import time

from giskardpy import logging
from giskardpy.exceptions import PathCollisionException, InsolvableException
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.utils import closest_point_constraint_violated
import numpy as np

class WiggleCancel(GiskardBehavior):
    def __init__(self, name):
        super(WiggleCancel, self).__init__(name)
        self.wiggle_precision1 = self.get_god_map().safe_get_data(identifier.wiggle_precision_threshold)

    def initialise(self):
        self.past_joint_states = set()
        super(WiggleCancel, self).initialise()

    def update(self):
        current_js = self.get_god_map().safe_get_data(identifier.joint_states)
        hz = self.get_god_map().safe_get_data(identifier.sample_period)
        current_time = self.get_god_map().safe_get_data(identifier.time_identifier) * hz
        rounded_js = self.round_js(current_js)
        # TODO make 1 a parameter
        # if current_time >= 18:
        #     self.check_fft()
        if current_time >= 1 and rounded_js in self.past_joint_states:
            current_max_joint_vel = np.abs([v.velocity for v in current_js.values()]).max()
            # TODO this threshold should depend on the joint type
            if current_max_joint_vel < 0.25:
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


    def check_fft(self):
        traj = self.get_god_map().safe_get_data(identifier.trajectory_identifier)
        hz = self.get_god_map().safe_get_data(identifier.sample_period)
        current_time = self.get_god_map().safe_get_data(identifier.time_identifier)
        last_sec = []
        keys = traj._points[0.0].keys()
        for time in range(int(current_time-1/hz), current_time):
            # rounded_time = np.round(time / hz) * hz
            point = traj._points[time]
            positions = [point[key].position for key in keys]
            last_sec.append(positions)
        last_sec = np.array(last_sec)

class MaxTrajLength(GiskardBehavior):
    pass


class CollisionCancel(GiskardBehavior):
    def __init__(self, name):
        self.collision_time_threshold = self.get_god_map().safe_get_data(identifier.collision_time_threshold)
        super(CollisionCancel, self).__init__(name)

    def update(self):
        time = self.get_god_map().safe_get_data(identifier.time_identifier)
        if time >= self.collision_time_threshold:
            cp = self.get_god_map().safe_get_data(identifier.closest_point)
            if cp is not None and closest_point_constraint_violated(cp, tolerance=1):
                self.raise_to_blackboard(PathCollisionException(
                    u'robot is in collision after {} seconds'.format(self.collision_time_threshold)))
                return Status.SUCCESS
        return Status.FAILURE
