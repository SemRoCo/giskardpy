from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class KinSimPlugin(GiskardBehavior):
    @profile
    def initialise(self):
        self.sample_period = self.god_map.get_data(identifier.sample_period)
        def f(joint_symbol):
            return self.god_map.expr_to_key[joint_symbol][-2]
        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    @record_time
    @profile
    def update(self):
        next_cmds = self.god_map.get_data(identifier.qp_solver_solution)
        self.world.update_state(next_cmds, self.sample_period)
        return Status.RUNNING
