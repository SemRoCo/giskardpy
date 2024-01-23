from py_trees import Status

from giskardpy.god_map import god_map
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time, catch_and_raise_to_blackboard
from giskardpy.utils.utils import is_running_in_pytest


class RealKinSimPlugin(GiskardBehavior):
    last_time: float
    print_warning = is_running_in_pytest()

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
        next_cmds = god_map.qp_solver_solution
        dt = next_time - self.last_time
        if dt > god_map.qp_controller_config.sample_period:
            # if self.print_warning:
            #     logging.logwarn(f'dt is larger than sample period of the MPC! '
            #                     f'{dt:.5f} > {god_map.qp_controller_config.sample_period}. ')
            dt = god_map.qp_controller_config.sample_period
        god_map.world.update_state(next_cmds, dt)
        self.last_time = next_time
        return Status.RUNNING
