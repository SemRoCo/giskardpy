from time import time

import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.utils import logging
from giskardpy.tree.plugin import GiskardBehavior


# fast

def make_velocity_threshold(god_map,
                            min_translation_cut_off=0.003,
                            min_rotation_cut_off=0.01,
                            max_translation_cut_off=0.01,
                            max_rotation_cut_off=0.06):
    joint_convergence_threshold = god_map.get_data(identifier.joint_convergence_threshold)
    controlled_joints = god_map.get_data(identifier.controlled_joints)
    world = god_map.get_data(identifier.world)
    thresholds = []
    for joint_name in controlled_joints:
        velocity_limit, _ = world.get_joint_velocity_limits(joint_name)
        if velocity_limit is None:
            velocity_limit = 1
        velocity_limit *= joint_convergence_threshold
        if world.is_joint_prismatic(joint_name):
            velocity_limit = min(max(min_translation_cut_off, velocity_limit), max_translation_cut_off)
        elif world.is_joint_rotational(joint_name):
            velocity_limit = min(max(min_rotation_cut_off, velocity_limit), max_rotation_cut_off)
        thresholds.append(velocity_limit)
    return np.array(thresholds)


class GoalReachedPlugin(GiskardBehavior):
    def __init__(self, name):
        super(GoalReachedPlugin, self).__init__(name)
        self.window_size = self.get_god_map().get_data(identifier.GoalReached_window_size)
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)

        self.above_threshold_time = 0

        self.thresholds = make_velocity_threshold(self.get_god_map())
        self.number_of_controlled_joints = len(self.thresholds)

    @profile
    def update(self):
        planning_time = self.get_god_map().get_data(identifier.time)
        controlled_joints = self.get_god_map().get_data(identifier.controlled_joints)['/pr2_a']
        if planning_time - self.above_threshold_time >= self.window_size:
            velocities = np.array([self.world.state[j].velocity for j in controlled_joints])
            below_threshold = np.all(np.abs(velocities[:self.number_of_controlled_joints]) < self.thresholds)
            if below_threshold:
                logging.loginfo(u'Found goal trajectory with length {:.3f}s in {:.3f}s'.format(planning_time * self.sample_period,
                                                                                       self.get_runtime()))
                return Status.SUCCESS
        return Status.RUNNING
