from py_trees import Status

import giskardpy.identifier as identifier
from giskardpy.data_types import KeyDefaultDict
from giskardpy.god_map_user import GodMap
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class KinSimPlugin(GiskardBehavior):
    @profile
    def initialise(self):
        def f(joint_symbol):
            return GodMap.god_map.expr_to_key[joint_symbol][-2]
        self.symbol_to_joint_map = KeyDefaultDict(f)
        super().initialise()

    @record_time
    @profile
    def update(self):
        next_cmds = GodMap.god_map.get_data(identifier.qp_solver_solution)
        GodMap.get_world().update_state(next_cmds, GodMap.get_sample_period())
        GodMap.get_world().notify_state_change()
        return Status.SUCCESS
