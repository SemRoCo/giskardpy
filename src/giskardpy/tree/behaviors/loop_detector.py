from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging


class LoopDetector(GiskardBehavior):
    past_joint_states: set

    def __init__(self, name):
        self.precision = 4
        self.window_size = 21
        super(LoopDetector, self).__init__(name)

    def initialise(self):
        super(LoopDetector, self).initialise()
        self.past_joint_states = set()

    @profile
    def update(self):
        current_js = self.get_god_map().get_data(identifier.joint_states)
        planning_time = self.get_god_map().get_data(identifier.time)
        rounded_js = self.round_js(current_js)
        if planning_time >= self.window_size and rounded_js in self.past_joint_states:
            sample_period = self.get_god_map().get_data(identifier.sample_period)
            logging.loginfo('found loop, stopped planning.')
            logging.loginfo('found goal trajectory with length {:.3f}s in {:.3f}s'.format(planning_time * sample_period,
                                                                                          self.get_runtime()))
            return Status.SUCCESS
        self.past_joint_states.add(rounded_js)
        return Status.RUNNING

    def round_js(self, js):
        """
        :param js: joint_name -> SingleJointState
        :type js: dict
        :return: a sequence of all the rounded joint positions
        :rtype: tuple
        """
        # FIXME weird non deterministic error, happens because pluginbehavior is not stopped fast enough?
        return tuple(round(x.position, self.precision) for x in js.values())
