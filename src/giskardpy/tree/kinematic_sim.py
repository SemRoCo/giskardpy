from collections import defaultdict
from copy import deepcopy

from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import JointStates, KeyDefaultDict
from giskardpy.tree.plugin import GiskardBehavior


class KinSimPlugin(GiskardBehavior):
    def __init__(self, name):
        super(KinSimPlugin, self).__init__(name)

    def initialise(self):
        self.sample_period = self.god_map.get_data(identifier.sample_period)
        def f(joint_symbol):
            return self.god_map.expr_to_key[joint_symbol][-2]
        self.symbol_to_joint_map = KeyDefaultDict(f)
        super(KinSimPlugin, self).initialise()

    @profile
    def update(self):
        next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        # next_state = JointStates()

        # for joint_name in self.world.controlled_joints:
        #     self.world.joints[joint_name].update_state(next_cmds, self.sample_period)
        # if next_cmds:
        for joint_symbol in next_cmds[0]:
            joint_name = self.symbol_to_joint_map[joint_symbol]
            self.world.joints[joint_name].update_state(next_cmds, self.sample_period)
                    # if i == 0:
                    #     next_state[joint_name].position = self.world.state[joint_name].position + cmd * self.sample_period
                    # next_state[joint_name].set_derivative(i + 1, cmd)
        # for joint_name in self.world.movable_joints_as_set.difference(set(self.symbol_to_joint_map.values())):
            # FIXME might want to copy vel etc too, but I don't think it it necessary
            # next_state[joint_name].position = self.world.state[joint_name].position
        # self.world.state = next_state
        self.world.notify_state_change()
        return Status.RUNNING
