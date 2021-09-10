from collections import OrderedDict

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates
from giskardpy.tree.plugin import GiskardBehavior


class KinSimPlugin(GiskardBehavior):
    def __init__(self, name):
        super(KinSimPlugin, self).__init__(name)

    def initialise(self):
        self.sample_period = self.god_map.get_data(identifier.sample_period)
        super(KinSimPlugin, self).initialise()

    @profile
    def update(self):
        next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        if next_cmds:
            for i, cmds in enumerate(next_cmds):
                for joint_symbol, cmd in cmds.items():
                    joint_name = self.god_map.expr_to_key[joint_symbol][-2]
                    if i == 0:
                        self.world.state[joint_name].position += cmd * self.sample_period
                    self.world.state[joint_name].set_derivative(i+1, cmd)
        self.world.soft_reset()
        return Status.RUNNING
