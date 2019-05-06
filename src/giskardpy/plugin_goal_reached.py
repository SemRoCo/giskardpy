import numpy as np
from time import time

from py_trees import Status

from giskardpy.identifier import time_identifier, js_identifier
from giskardpy.plugin import GiskardBehavior


class GoalReachedPlugin(GiskardBehavior):
    def __init__(self, name, joint_convergence_threshold):
        super(GoalReachedPlugin, self).__init__(name)
        self.joint_convergence_threshold = joint_convergence_threshold

    def update(self):
        current_js = self.get_god_map().safe_get_data(js_identifier)
        planning_time = self.get_god_map().safe_get_data(time_identifier)
        # TODO make 1 a parameter
        if planning_time >= 1:
            if np.abs([v.velocity for v in current_js.values()]).max() < self.joint_convergence_threshold:
                print(u'found goal trajectory with length {}s in {}s'.format(planning_time,
                                                                             time() - self.get_blackboard().runtime))
                return Status.SUCCESS
        return Status.RUNNING
