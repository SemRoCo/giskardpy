from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.god_map_interpreter import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class KinSimPlugin(GiskardBehavior):
    @profile
    def initialise(self):
        def f(joint_symbol):
            return god_map.expr_to_key[joint_symbol][-2]
        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    @record_time
    @profile
    def update(self):
        next_cmds = god_map.get_data(identifier.qp_solver_solution)
        god_map.world.update_state(next_cmds, god_map.sample_period)
        god_map.world.notify_state_change()
        return Status.SUCCESS
