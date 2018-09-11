import numpy as np

from py_trees import Status

from giskardpy.plugin import NewPluginBase


class GoalReachedPlugin(NewPluginBase):
    def __init__(self, joint_state_identifier, time_identifier, joint_convergence_threshold):
        super(GoalReachedPlugin, self).__init__()
        self.joint_state_identifier = joint_state_identifier
        self.time_identifier = time_identifier
        self.joint_convergence_threshold = joint_convergence_threshold

    def setup(self):
        super(GoalReachedPlugin, self).setup()

    def initialize(self):
        super(GoalReachedPlugin, self).initialize()

    def update(self):
        current_js = self.god_map.safe_get_data([self.joint_state_identifier])
        time = self.god_map.safe_get_data([self.time_identifier])
        # TODO make 1 a parameter
        if time >= 1:
            if np.abs([v.velocity for v in current_js.values()]).max() < self.joint_convergence_threshold:
                print(u'done')
                return Status.SUCCESS
        return Status.RUNNING