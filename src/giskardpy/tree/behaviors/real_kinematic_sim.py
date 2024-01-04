from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard


class RealKinSimPlugin(GiskardBehavior):
    last_time: float

    def initialise(self):
        self.last_time = None
        self.start_time = god_map.tracking_start_time

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        next_time = god_map.time
        if next_time <= 0.0 or self.last_time is None:
            self.last_time = next_time
            return Status.RUNNING
        # if self.last_time is None:
        next_cmds = god_map.qp_solver_solution
        dt = next_time - self.last_time
        god_map.world.update_state(next_cmds, dt)
        self.last_time = next_time
        # god_map.get_world().notify_state_change()
        return Status.RUNNING
