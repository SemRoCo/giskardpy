from collections import OrderedDict

import numpy as np
from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import SingleJointState
from giskardpy.plugin import GiskardBehavior


class KinSimPlugin(GiskardBehavior):
    def __init__(self, name):
        """
        :type js_identifier: str
        :type next_cmd_identifier: str
        :type time_identifier: str
        :param sample_period: the time difference in s between each step.
        :type sample_period: float
        """
        super(KinSimPlugin, self).__init__(name)

    def initialise(self):
        self.sample_period = self.get_god_map().get_data(identifier.sample_period)
        super(KinSimPlugin, self).initialise()

    @profile
    def update(self):
        next_cmds = self.get_god_map().get_data(identifier.qp_solver_solution)
        current_js = self.get_god_map().get_data(identifier.joint_states)
        next_js = None
        if next_cmds:
            next_js = OrderedDict()
            for key, sjs in current_js.items():
                joint_name = str(self.get_robot().get_joint_position_symbol(key))
                vel_cmds = next_cmds[0]
                if joint_name in vel_cmds:
                    cmd = vel_cmds[joint_name]
                    derivative_cmds = [x[joint_name] for x in next_cmds]
                else:
                    cmd = 0.0
                    derivative_cmds = []
                next_js[key] = SingleJointState(sjs.name,
                                                sjs.position + cmd * self.sample_period,
                                                *derivative_cmds)
        if next_js is not None:
            self.get_god_map().set_data(identifier.joint_states, next_js)
        else:
            self.get_god_map().set_data(identifier.joint_states, current_js)
        self.get_god_map().set_data(identifier.last_joint_states, current_js)
        return Status.RUNNING
