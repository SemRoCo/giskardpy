import numpy as np
from time import time

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy import logging


class GoalReachedPlugin(GiskardBehavior):
    def __init__(self, name):
        super(GoalReachedPlugin, self).__init__(name)
        self.joint_convergence_threshold = self.get_god_map().safe_get_data(identifier.joint_convergence_threshold)

    def update(self):
        current_js = self.get_god_map().safe_get_data(identifier.joint_states)
        hz = self.get_god_map().safe_get_data(identifier.sample_period)
        planning_time = self.get_god_map().safe_get_data(identifier.time_identifier) * hz
        # TODO make 1 a parameter
        if planning_time >= 1:
            if np.abs([v.velocity for v in current_js.values()]).max() < self.joint_convergence_threshold:
                logging.loginfo(u'found goal trajectory with length {}s in {}s'.format(planning_time,
                                                                             time() - self.get_blackboard().runtime))
                return Status.SUCCESS
        return Status.RUNNING
