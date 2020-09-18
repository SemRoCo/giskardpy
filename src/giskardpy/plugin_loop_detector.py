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


class LoopDetector(GiskardBehavior):
    def __init__(self, name):
        self.precision = 3
        self.window_size = 21
        super(LoopDetector, self).__init__(name)

    def initialise(self):
        super(LoopDetector, self).initialise()
        self.past_joint_states = set()

    def update(self):
        current_js = self.get_god_map().get_data(identifier.joint_states)
        time = self.get_god_map().get_data(identifier.time)
        rounded_js = self.round_js(current_js)
        if time >= self.window_size and rounded_js in self.past_joint_states:
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



