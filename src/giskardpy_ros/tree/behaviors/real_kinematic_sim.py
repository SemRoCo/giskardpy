from line_profiler import profile
from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy_ros.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time
from giskardpy_ros.tree.blackboard_utils import catch_and_raise_to_blackboard
from giskardpy.utils.utils import is_running_in_pytest


class RealKinSimPlugin(GiskardBehavior):
    last_time: float
    print_warning = is_running_in_pytest()

    def initialise(self):
        self.last_time = None

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        next_time = god_map.time
        if next_time <= 0.0 or self.last_time is None:
            self.last_time = next_time
            return Status.RUNNING
        next_cmds = god_map.qp_solver_solution
        dt = next_time - self.last_time
        if dt > god_map.qp_controller.mpc_dt:
            dt = god_map.qp_controller.mpc_dt
        god_map.world.update_state(next_cmds, dt, max_derivative=god_map.qp_controller.max_derivative)
        self.last_time = next_time
        return Status.RUNNING
