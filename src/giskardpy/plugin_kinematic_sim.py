from collections import OrderedDict

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
        vel_cmd, acc_cmds = self.get_god_map().get_data(identifier.cmd)
        current_js = self.get_god_map().get_data(identifier.joint_states)
        next_js = None
        if vel_cmd:
            next_js = OrderedDict()
            for joint_name, sjs in current_js.items():
                if joint_name in vel_cmd:
                    cmd = vel_cmd[joint_name]
                    acc_cmd = acc_cmds[joint_name]
                else:
                    cmd = 0.0
                    acc_cmd = 0.0
                next_js[joint_name] = SingleJointState(name=sjs.name,
                                                       position=sjs.position + cmd * self.sample_period,
                                                       velocity=cmd,
                                                       acceleration=acc_cmd)
        if next_js is not None:
            self.get_god_map().set_data(identifier.joint_states, next_js)
        else:
            self.get_god_map().set_data(identifier.joint_states, current_js)
        self.get_god_map().set_data(identifier.last_joint_states, current_js)
        return Status.RUNNING
